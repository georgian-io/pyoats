TARGET_PATH="./data/skab"
mkdir -p "$TARGET_PATH"


svn export --force https://github.com/waico/SKAB/trunk/data $TARGET_PATH
