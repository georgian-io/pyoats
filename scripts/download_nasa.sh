#!/bin/sh

DOWNLOAD_LINK="https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
TEMP_NAME="temp"
TEMP_PATH="./$TEMP_NAME"
TEMP_FPATH="$TEMP_PATH/dataset.zip"
DATA_PATH="/data"

TARGET_PATH="./data/nasa"


# mkdir
mkdir -p "$TARGET_PATH"
mkdir -p "$TEMP_PATH"

# download file
echo Downloading .zip file...
curl "$DOWNLOAD_LINK" -o "$TEMP_FPATH"

# unzip
echo Unzipping...
unzip -q "$TEMP_FPATH" -d "$TEMP_PATH"

# copy data
echo Copying data...
cp -a "$TEMP_PATH$DATA_PATH/." "$TARGET_PATH"
svn export --force https://github.com/khundman/telemanom/trunk/labeled_anomalies.csv $TARGET_PATH

# clean up
echo Cleaning up...
rm -r $TEMP_PATH
rm -r "$TARGET_PATH/2018-05-19_15.00.10"


