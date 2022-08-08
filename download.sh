kaggle competitions download -c feedback-prize-2021
kaggle competitions download -c feedback-prize-effectiveness

unzip feedback-prize-effectiveness.zip -d ./2022
unzip feedback-prize-2021.zip -d ./2021

rm -rf feedback-prize-effectiveness.zip
rm -rf feedback-prize-2021.zip

python process.py
python process_pseudo.py
python cv_split.py