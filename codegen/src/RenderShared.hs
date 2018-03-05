{-# LANGUAGE NamedFieldPuns #-}
module RenderShared
  ( makeModule
  , renderCHeaderFile

  , IsTemplate(..)

  , parseFile
  , cleanList
  ) where

import CodeGen.Prelude
import qualified Data.Text as T

import CodeGen.Types
import CodeGen.Render.Function (renderFunPtrSig, renderFunSig)
import CodeGenParse (thParser)
import ConditionalCases (checkFunction, signatureAliases)
import qualified CodeGen.Render.Haskell as Hs

makeModule
  :: LibType
  -> TextPath
  -> IsTemplate
  -> FilePath
  -> ModuleSuffix
  -> FileSuffix
  -> TemplateType
  -> [THFunction]
  -> HModule
makeModule a00 a0 a1 a2 a3 a4 a5 a6
  = HModule
  { modPrefix = a00
  , modExtensions = ["ForeignFunctionInterface"]
  , modImports = ["Foreign", "Foreign.C.Types", "THTypes", "Data.Word", "Data.Int"]
  , modTypeDefs = []
  , modOutDir = a0
  , modIsTemplate = a1
  , modHeader = a2
  , modSuffix = a3
  , modFileSuffix = a4
  , modTypeTemplate = a5
  , modBindings = a6
  }

-- makeTHModule :: TextPath -> IsTemplate -> FilePath -> ModuleSuffix -> FileSuffix -> TemplateType -> [THFunction] -> HModule
-- makeTHModule = makeModule TH

-- ----------------------------------------
-- helper data and functions for templating
-- ----------------------------------------

makePrefix :: Text -> Text
makePrefix templateType = "TH" <> templateType <> "Tensor"

renderExtension :: Text -> Text
renderExtension extension = "{-# LANGUAGE " <> extension <> " #-}"

renderExtensions :: [Text] -> Text
renderExtensions extensions = T.intercalate "\n" (extensions' <> [""])
 where
  extensions' :: [Text]
  extensions' = renderExtension <$> extensions

renderModule :: HModule -> Text
renderModule moduleSpec = "module " <> renderModuleName moduleSpec

renderExports :: [Text] -> Text
renderExports exports = T.intercalate "\n"
  [ ""
  , "  ( " <> T.intercalate "\n  , " exports
  , "  ) where"
  , ""
  , ""
  ]

renderImports :: [Text] -> Text
renderImports imports = T.intercalate "\n" (("import " <>) <$> imports) <> "\n\n"


-- TODO clean up redundancy of valid functions vs. functions in moduleSpec
renderFunctions :: HModule -> [THFunction] -> Text
renderFunctions m validFunctions =
  T.intercalate "\n\n"
    $  (renderFunSig'    <$> triple)
    <> (renderFunPtrSig' <$> triple)
 where
  renderFunSig'    = renderFunSig    (modIsTemplate m) ffiPrefix (modHeader m) (modTypeTemplate m)
  renderFunPtrSig' = renderFunPtrSig (modIsTemplate m) ffiPrefix (modHeader m) (modTypeTemplate m)

  ffiPrefix :: Text
  ffiPrefix = T.pack (show $ modPrefix m) <> Hs.type2SpliceReal (modTypeTemplate m) <> textSuffix (modSuffix m)

  triple :: [(Text, THType, [THArg])]
  triple = go <$> validFunctions
    where
      go :: THFunction -> (Text, THType, [THArg])
      go f = (funName f, funReturn f, funArgs f)

-- | Check for conditional templating of functions and filter function list
checkList :: [THFunction] -> TemplateType -> [THFunction]
checkList fList templateType = filter ((checkFunction templateType) . FunctionName . funName) fList

renderAll :: HModule -> Text
renderAll m
  =  renderExtensions (modExtensions m)
  <> renderModule m
  <> renderExports exportFunctions
  <> renderImports (modImports m)
  <> renderFunctions m validFunctions
  where
    validFunctions :: [THFunction]
    validFunctions = checkList (modBindings m) (modTypeTemplate m)

    fun2name :: Text -> THFunction -> Text
    fun2name p = (\f -> p <> "_" <> f) . funName

    exportFunctions :: [Text]
    exportFunctions
      =  (fmap (fun2name "c") validFunctions)
      <> (fmap (fun2name "p") validFunctions)

renderCHeaderFile
  :: [THFunction] -> (TemplateType -> [THFunction] -> HModule) -> TemplateType -> IO ()
renderCHeaderFile parsedBindings makeConfig templateType = do
  putStrLn $ "Writing " <> T.unpack filename
  writeFile (outDir ++ T.unpack filename) (T.unpack . renderAll $ modSpec)
 where
  modSpec :: HModule
  modSpec = makeConfig templateType parsedBindings

  filename :: Text
  filename = renderModuleName modSpec <> ".hs"

  outDir :: String
  outDir = T.unpack (textPath $ modOutDir modSpec)

renderModuleName :: HModule -> Text
renderModuleName HModule{modPrefix, modTypeTemplate, modFileSuffix}
  = T.pack (show modPrefix) <> (Hs.type2SpliceReal modTypeTemplate) <> textFileSuffix modFileSuffix

-- ----------------------------------------
-- Execution
-- ----------------------------------------

-- |Remove If list was returned, extract non-Nothing values, o/w empty list
cleanList :: Either (ParseError Char Void) [Maybe THFunction] -> [THFunction]
cleanList = either (const []) catMaybes

parseFile :: CodeGenType -> String -> IO [THFunction]
parseFile cgt file = do
  putStrLn $ "\nParsing " ++ file ++ " ... "
  res <- parseFromFile (thParser cgt) file
  pure $ cleanList res
 where
  parseFromFile
    :: Parser [Maybe THFunction]
    -> String
    -> IO (Either (ParseError Char Void) [Maybe THFunction])
  parseFromFile p file = runParser p file <$> readFile file

