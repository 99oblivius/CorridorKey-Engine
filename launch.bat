@echo off
REM CorridorKey — quick launcher.
REM Forwards all arguments to tools\cli.bat.
REM Usage: launch.bat [subcommand] [options]
REM   launch.bat                          — TUI
REM   launch.bat inference C:\path\to\clips — headless keying
REM   launch.bat --help

call "%~dp0tools\cli.bat" %*
