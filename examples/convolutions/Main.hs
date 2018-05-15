module Main where

import qualified Conv1d
import qualified Conv2d
import qualified ReLU
import qualified MaxPooling

main :: IO ()
main = do
  Conv1d.main
  Conv2d.main
  ReLU.main
  MaxPooling.main

