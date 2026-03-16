@echo off
REM CorridorKey — quick launcher.
REM Forwards all arguments to tools\cli.bat.
REM Usage: launch.bat [subcommand] [options]
REM   launch.bat wizard "C:\path\to\clips"
REM   launch.bat run-inference
REM   launch.bat --help

call "%~dp0tools\cli.bat" %*
