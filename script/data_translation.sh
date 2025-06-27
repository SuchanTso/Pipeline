sh check_predefine.sh $1

exit_code=$?

if [ $exit_code -eq 0 ]; then
    python src/dataset/translate.py -i "$1" -o "$2" --config "config/translate.yaml"
else
    echo "Predefine values are incorrect."
    exit 1
fi