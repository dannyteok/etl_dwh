#!/bin/bash

## Logging function with date-timestamp. Exit on ERROR
function Log {
echo "$(date) [$1] "$2"" 1>&2
if [[ "$1" = "ERROR" ]]; then
  exit 9
fi
}
