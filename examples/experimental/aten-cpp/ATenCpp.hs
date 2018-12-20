{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
module Main where

import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C

C.context C.cppCtx
C.include "<ATen/ATen.h>"
C.include "<iostream>"
C.include "<stdexcept>"

main :: IO ()
main = do
  [C.block| void {
    std::cout << at::native::ones({2, 3}, at::kInt) << std::endl;
  } |]
  print "hello"

