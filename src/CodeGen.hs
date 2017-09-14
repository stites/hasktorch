{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad (void)
import Data.Monoid ((<>))
import Data.Maybe
import Data.Void
import Data.Text
import Data.Text as T
import Text.Megaparsec
import Text.Megaparsec.Char
import Text.Megaparsec.Expr
import qualified Text.Megaparsec.Char.Lexer as L
import Prelude as P
import Text.Show.Pretty

import CodeParse

-- ----------------------------------------
-- Types for rendering output
-- ----------------------------------------

data TemplateType = GenByte
                  | GenChar
                  | GenDouble
                  | GenFloat
                  | GenHalf
                  | GenInt
                  | GenLong
                  | GenShort deriving Show

data HModule = HModule {
  modHeader :: Text,
  modPrefix :: Text,
  modTypeTemplate :: TemplateType,
  modSuffix :: Text,
  modExtensions :: [Text],
  modImports :: [Text],
  modTypeDefs :: [(Text, Text)],
  modBindings :: [THFunction]
  } deriving Show

data TypeCategory = ReturnValue | FunctionParam

-- ----------------------------------------
-- Rendering
-- ----------------------------------------

makePrefix :: Text -> Text
makePrefix templateType = "TH" <> templateType <> "Tensor"

-- #define Real [X]
-- spliced text to use for function names
type2SpliceReal :: TemplateType -> Text
type2SpliceReal GenByte   = "Byte"
type2SpliceReal GenChar   = "Byte"
type2SpliceReal GenDouble = "Double"
type2SpliceReal GenFloat  = "Float"
type2SpliceReal GenHalf   = "Half"
type2SpliceReal GenInt    = "Int"
type2SpliceReal GenLong   = "Long"
type2SpliceReal GenShort  = "Short"

-- #define real [X]
type2real :: TemplateType -> Text
type2real GenByte   = "unsigned char"
type2real GenChar   = "char"
type2real GenDouble = "double"
type2real GenFloat  = "float"
type2real GenHalf   = "THHalf"
type2real GenInt    = "int"
type2real GenLong   = "long"
type2real GenShort  = "short"

-- #define accreal [X]
type2accreal :: TemplateType -> Text
type2accreal GenByte   = "long"
type2accreal GenChar   = "long"
type2accreal GenDouble = "double"
type2accreal GenFloat  = "double"
type2accreal GenHalf   = "float"
type2accreal GenInt    = "long"
type2accreal GenLong   = "long"
type2accreal GenShort  = "long"

renderCType :: THType -> Text
renderCType THVoid = "void"
renderCType THDescBuff = "THDescBuff"
renderCType THTensorPtr = "THTensor *"
renderCType THTensorPtrPtr = "THTensor **"
renderCType THStoragePtr = "THStorage *"
renderCType THLongStoragePtr = "THLongStorage *"
renderCType THPtrDiff = "ptrdiff_t"
renderCType THLong = "long"
renderCType THInt = "int"
renderCType THChar = "char"
renderCType THRealPtr = "real *"
renderCType THReal = "real"
renderCType THAccRealPtr = "accreal *"
renderCType THAccReal = "accreal"

renderHaskellType :: TypeCategory -> TemplateType -> THType -> Maybe Text
renderHaskellType typeCat templateType THVoid =
  case typeCat of
    ReturnValue -> Just "IO ()"
    FunctionParam -> Nothing

renderHaskellType _ _ THDescBuff = Just "CTHDescBuff"

renderHaskellType _ templateType THTensorPtr =
  Just ("Ptr CTH" <> type2SpliceReal templateType)

renderHaskellType _ templateType THTensorPtrPtr =
  Just $ "Ptr (Ptr CTH" <> type2SpliceReal templateType <> "Tensor)"

renderHaskellType _ templateType THStoragePtr =
  Just $ "Ptr CTH" <> type2SpliceReal templateType <> "Storage"

renderHaskellType _ templateType THLongStoragePtr =
  Just $ "Ptr CTH" <> type2SpliceReal templateType <> "LongStorage"

renderHaskellType _ templateType THPtrDiff =
  Just "CTHFloatPtrDiff"

renderHaskellType _ templateType THLongPtr =
  Just "Ptr CLong"

renderHaskellType _ templateType THLong =
  Just "CLong"

renderHaskellType _ templateType THInt =
  Just "CInt"

renderHaskellType _ templateType THChar =
  Just "CChar"

renderHaskellType _ templateType THRealPtr =
  Just "Ptr " -- TODO

renderHaskellType _ templateType THReal =
  Just "[XXX]" -- TODO

renderHaskellType _ templateType THAccRealPtr =
  Just "Ptr [XXX]" -- TODO

renderHaskellType _ templateType THAccReal =
  Just "[XXX]" -- TODO

renderExtension :: Text -> Text
renderExtension extension = "{-# LANGUAGE " <> extension <> "#-}"

renderExtensions :: [Text] -> Text
renderExtensions extensions = T.intercalate "\n" (renderExtension <$> extensions)

renderModuleName :: HModule -> Text
renderModuleName HModule{..} =
  modPrefix <> (type2SpliceReal modTypeTemplate) <> modSuffix

renderModule :: HModule -> Text
renderModule moduleSpec =
  "module " <> (renderModuleName moduleSpec)

renderExports :: [Text] -> Text
renderExports exports = (" (\n    "
                         <> (T.intercalate ",\n    " exports)
                         <> ") where\n\n")

renderImports :: [Text] -> Text
renderImports imports = (T.intercalate "\n" (singleimport <$> imports)) <> "\n\n"
  where singleimport x = "import " <> x

renderFunName :: Text -> Text -> Text
renderFunName prefix name = prefix <> "_" <> name

renderFunSig :: TemplateType -> (Text, THType, [THArg]) -> Text
renderFunSig modTypeTemplate (name, retType, args) =
  ("foreign import ccall \"THTensor.h " <> name <> "\"\n"
   <> "  c_" <> name <> " :: "
   <> (T.intercalate " -> " $ catMaybes typeSignature)
   -- TODO : fromJust shouldn't fail, clean this up so it's not unsafe
   <> " -> " <> fromJust (renderHaskellType ReturnValue modTypeTemplate retType) <> "\n"
   <> "  -- " <> (T.intercalate " " nameSignature) <> " -> " <> (renderCType retType)
  )
  where
    typeVals = thArgType <$> args
    typeSignature = renderHaskellType FunctionParam modTypeTemplate <$> typeVals
    nameSignature = thArgName <$> args

renderFunctions :: HModule -> Text
renderFunctions moduleSpec@HModule{..} =
  -- iteration over all functions
  intercalate "\n\n" ((renderFunSig typeTemplate)
                      <$> (P.zip3 funNames retTypes args) )
  where
    modulePrefix = (renderModuleName moduleSpec) <> "_"
    funNames = (mappend modulePrefix) <$> funName <$> modBindings
    retTypes = funReturn <$> modBindings
    args = funArgs <$> modBindings
    typeTemplate = modTypeTemplate

renderAll :: HModule -> Text
renderAll spec =
    renderModule spec
    <> renderExports exportFunctions
    <> renderImports (modImports spec)
    <> renderFunctions spec
  where
    prefix = makePrefix . type2SpliceReal . modTypeTemplate $ spec
    bindings = modBindings spec
    exportFunctions =
      (renderFunName ("c_" <> renderModuleName spec)
       <$> (fmap funName (modBindings spec)))

-- ----------------------------------------
-- Execution
-- ----------------------------------------

parseFromFile p file = runParser p file <$> readFile file

cleanList :: Either (ParseError Char Void) [Maybe THFunction] -> [THFunction]
cleanList (Left _) = []
cleanList (Right lst) = fromJust <$> (P.filter f lst)
  where
    f Nothing = False
    f (Just _) = True

makeModule typeTemplate bindings =
   HModule {
        modHeader = "Tensor.h" 
        modPrefix = "TH",
        modTypeTemplate = typeTemplate,
        modSuffix = "Tensor",
        modExtensions = ["ForeignFunctionInterface"],
        modImports = ["Foreign", "Foreign.C.Types", "THTypes"],
        modTypeDefs = [],
        modBindings = bindings
  }

renderTensorFile templateType parsedBindings = do
  putStrLn $ "Writing " <> T.unpack filename
  writeFile ("./render/" ++ T.unpack filename) (T.unpack . renderAll $ modSpec)
  where modSpec = makeModule templateType parsedBindings
        filename = (renderModuleName modSpec) <> ".hs"

genTypes = [GenByte, GenChar,
            GenDouble, GenFloat, GenHalf,
            GenInt, GenLong, GenShort] :: [TemplateType]

runTensor = do
  parsedBindings <- testFile "vendor/torch7/lib/TH/generic/THTensor.h"
  putStrLn "First 3 signatures"
  putStrLn $ ppShow (P.take 3 parsedBindings)
  mapM_ (\x -> renderTensorFile x parsedBindings) genTypes

testString inp = case (parse thFile "" inp) of
  Left err -> putStrLn (parseErrorPretty err)
  Right val -> putStrLn $ (ppShow val)

testFile file = do
  res <- parseFromFile thFile file
  pure $ cleanList res

test1 = do
  testString ex1
  where
    ex1 = "skip this garbage line line\n" <>
     "TH_API void THTensor_(setFlag)(THTensor *self,const char flag);" <>
     "another garbage line ( )@#R @# 324 32"

main = do
  runTensor
  putStrLn "Done"
