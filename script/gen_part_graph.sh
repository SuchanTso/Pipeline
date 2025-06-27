input_file="$2"

# 判断是否传入了参数
if [ -z "$input_file" ]; then
  echo "请传入一个文件名作为参数"
  exit 1
fi

# 提取文件名的主干和后缀
base_name="${input_file%.*}"     # 去掉最后一个 . 及之后的内容，得到 xx
extension="${input_file##*.}"   # 取最后一个 . 之后的内容，得到 a

# 构造新文件名
new_file="${base_name}_temp.${extension}"
touch "$new_file"  # 创建一个空文件

sh data_translation.sh $2 $new_file

exit_code=$?

if [ $exit_code -eq 0 ]; then
    python src/dataset/extract_nodes.py -p $1 -w $new_file -o $3
else
    echo "err ocurrs when translation."
    exit 1
fi
