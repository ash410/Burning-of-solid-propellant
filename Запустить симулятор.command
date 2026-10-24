#!/bin/zsh

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR" || exit 1
export MPLCONFIGDIR="$SCRIPT_DIR/.mplconfig"

python3 "$SCRIPT_DIR/app.py"
status=$?

if (( status != 0 )); then
  echo
  echo "Не удалось запустить симулятор (код $status)."
  echo "Установите зависимости командой: python3 -m pip install -r requirements.txt"
  echo
  read "?Нажмите Enter, чтобы закрыть окно..."
fi

exit $status
