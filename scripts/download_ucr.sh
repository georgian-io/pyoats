#!/bin/sh

DOWNLOAD_LINK="https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip"
TEMP_NAME="temp"
TEMP_PATH="./$TEMP_NAME"
TEMP_FPATH="$TEMP_PATH/dataset.zip"
DATA_PATH="/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData"

TARGET_PATH="./data/ucr"

cd ..

# mkdir
if [ ! -d "$TEMP_PATH" ]
    then
        echo making temp directory...
        mkdir "$TEMP_NAME"
fi

# download file
echo Downloading .zip file...
curl "$DOWNLOAD_LINK" -o "$TEMP_FPATH"

# unzip
echo Unzipping...
unzip -q "$TEMP_FPATH" -d "$TEMP_PATH"

# copy data
echo Copying data...
cp -a "$TEMP_PATH$DATA_PATH/." "$TARGET_PATH"

# clean up
echo Cleaning up...
rm -r $TEMP_PATH


