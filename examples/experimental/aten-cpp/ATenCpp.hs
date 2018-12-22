-------------------------------------------------------------------------------
-- |
-- Module    :  ATenCpp
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Example usage of ATen from the readme with inline-c-cpp:
--
-- @
--   using namespace at; // assumed in the following
--
--   Tensor d = CPU(kFloat).ones({3, 4});
--   Tensor r = CPU(kFloat).zeros({3,4})
--   for(auto i = 0; i < 100000; i++) {
--     r = r.add(d);
--     // equivalently
--     r = r + d;
--     // or
--     r += d;
--   }
-- @
-------------------------------------------------------------------------------
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import Control.Exception.Safe
import Control.Monad
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import Language.C.Inline.Cpp
import Language.C.Inline.Cpp.Exceptions

C.context C.cppCtx
-- The equivalent of baseCtx for C++. It specifies the .cpp file extension for the C file, so that g++ will decide to build C++ instead of C. See the .cabal test target for an example on how to build.

C.using "namespace at"
-- Emits an using directive, e.g. @C.using "namespace std"@ ==> using namespace std


{-
A module that contains exception-safe equivalents of inline-c QuasiQuoters.


data CppException        -- An exception thrown in C++ code.
  = CppStdException String
  | CppOtherException
  deriving (Eq, Ord, Show, Exception)

throwBlock :: QuasiQuoter
  Like tryBlock, but will throw CppExceptions rather than returning them in an Either

tryBlock :: QuasiQuoter
  Similar to block, but C++ exceptions will be caught and the result is (Either CppException value). The return type must be void or constructible with {}. Using this will automatically include exception, cstring and cstdlib.

catchBlock :: QuasiQuoter
  Variant of throwBlock for blocks which return void.
-}

main :: IO ()
main = print "hello"

--- how to use inline-c-cpp
{-

C.context C.cppCtx

C.include "<iostream>"
C.include "<stdexcept>"

main :: IO ()
main = Hspec.hspec $ do
Hspec.describe "Basic C++" $ do
Hspec.it "Hello World" $ do
      let x = 3
[C.block| void {
          std::cout << "Hello, world!" << $(int x) << std::endl;
        } |]

  Hspec.describe "Exception handling" $ do
Hspec.it "std::exceptions are caught" $ do
result <- try [C.catchBlock|
        throw std::runtime_error("C++ error message");
        |]

      result `Hspec.shouldBe` Left (C.CppStdException "C++ error message")

Hspec.it "non-exceptions are caught" $ do
result <- try [C.catchBlock|
        throw 0xDEADBEEF;
        |]

      result `Hspec.shouldBe` Left C.CppOtherException

Hspec.it "catch without return (pure)" $ do
result <- [C.tryBlock| void {
          throw std::runtime_error("C++ error message");
        }
        |]

      result `Hspec.shouldBe` Left (C.CppStdException "C++ error message")

Hspec.it "try and return without throwing (pure)" $ do
result <- [C.tryBlock| int {
          return 123;
        }
        |]

      result `Hspec.shouldBe` Right 123

Hspec.it "return maybe throwing (pure)" $ do
result <- [C.tryBlock| int {
          if(1) return 123;
          else throw std::runtime_error("C++ error message");
        }
        |]

      result `Hspec.shouldBe` Right 123

Hspec.it "return definitely throwing (pure)" $ do
result <- [C.tryBlock| int {
          if(0) return 123;
          else throw std::runtime_error("C++ error message");
        }
        |]

      result `Hspec.shouldBe` Left (C.CppStdException "C++ error message")

Hspec.it "catch without return (pure)" $ do
result <- [C.tryBlock| void {
          throw std::runtime_error("C++ error message");
        }
        |]

      result `Hspec.shouldBe` Left (C.CppStdException "C++ error message")

Hspec.it "try and return without throwing (throw)" $ do
result :: Either C.CppException C.CInt <- try [C.throwBlock| int {
          return 123;
        }
        |]

      result `Hspec.shouldBe` Right 123

Hspec.it "return maybe throwing (throw)" $ do
result :: Either C.CppException C.CInt <- try [C.throwBlock| int {
          if(1) return 123;
          else throw std::runtime_error("C++ error message");
        }
        |]

      result `Hspec.shouldBe` Right 123

Hspec.it "return definitely throwing (throw)" $ do
result <- try [C.throwBlock| int {
          if(0) return 123;
          else throw std::runtime_error("C++ error message");
        }
        |]

      result `Hspec.shouldBe` Left (C.CppStdException "C++ error message")

Hspec.it "catch without return (throw)" $ do
result <- try [C.throwBlock| void {
          throw std::runtime_error("C++ error message");
        }
        |]

      result `Hspec.shouldBe` Left (C.CppStdException "C++ error message")

    Hspec.it "code without exceptions works normally" $ do
result :: Either C.CppException C.CInt <- try $ C.withPtr_ $ \resPtr -> [C.catchBlock|
          *$(int* resPtr) = 0xDEADBEEF;
        |]

result `Hspec.shouldBe` Right 0xDEADBEEF
-}
