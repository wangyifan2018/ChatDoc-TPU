pip3 install dfss --upgrade
python3 -m dfss --url=share@sophgo.com:chatdoc/nltk_data.zip
python3 -m dfss --url=share@sophgo.com:chatdoc/models.zip
unzip models.zip
rm models.zip
unzip nltk_data.zip
mv nltk_data.zip ~