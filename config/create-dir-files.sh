cd $HOME/data/rtts/train/JPEGImages/ && ls -d1 $PWD/* > train.txt && mv train.txt $HOME/data/rtts/

cd $HOME/data/rtts/test/JPEGImages/ && ls -d1 $PWD/* > test.txt && mv test.txt $HOME/data/rtts/

cd $HOME/data/rtts/valid/JPEGImages/ && ls -d1 $PWD/* > valid.txt && mv valid.txt $HOME/data/rtts/
