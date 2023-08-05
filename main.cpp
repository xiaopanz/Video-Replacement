#include <QApplication>
#include <QPushButton>
#include <QSoundEffect>
#include <QFileDialog>
#include <QString>
#include <QThread>
#include <QLabel>
#include <unistd.h>
#include "WavReader.h"
int main(int argc, char *argv[])
{

    QApplication a(argc, argv);
//    QString originalFileName = QFileDialog::getOpenFileName(nullptr, "open original wav file", " ",  "wav(*.wav);;Allfile(*.*)");
//    QString replaceFileName1 = QFileDialog::getOpenFileName(nullptr, "open first replace wav file", " ",  "wav(*.wav);;Allfile(*.*)");
//    QString replaceFileName2 = QFileDialog::getOpenFileName(nullptr, "open second replace wav file", " ",  "wav(*.wav);;Allfile(*.*)");


    // dataset1
    QString originalFileName = "data/dataset1/Videos/data_test1.wav";
    QString replaceFileName1 = "data/dataset1/Ads/Subway_Ad_15s.wav";
    QString replaceFileName2 = "data/dataset1/Ads/Starbucks_Ad_15s.wav";
    WavReader reader(originalFileName);
    reader.processOriginalFile();
    reader.replaceWav(replaceFileName1, 2400, 2849, replaceFileName2, 5550, 5999);
    QString placeToSave("output1.wav");
    reader.writeWav(placeToSave);

    /*
    // dataset2
    QString originalFileName = "data/dataset2/Videos/data_test2.wav";
    QString replaceFileName1 = "data/dataset2/Ads/nfl_Ad_15s.wav";
    QString replaceFileName2 = "data/dataset2/Ads/mcd_Ad_15s.wav";
    WavReader reader(originalFileName);
    reader.processOriginalFile();
    reader.replaceWav(replaceFileName1, 0, 449, replaceFileName2, 6000, 6449);
    QString placeToSave("output2.wav");
    reader.writeWav(placeToSave);
    */

    /*
    // dataset3
    QString originalFileName = "data/dataset3/Videos/data_test3.wav";
    QString replaceFileName1 = "data/dataset3/Ads/ae_ad_15s.wav";
    QString replaceFileName2 = "data/dataset3/Ads/hrc_ad_15s.wav";
    WavReader reader(originalFileName);
    reader.processOriginalFile();
    reader.replaceWav(replaceFileName1, 4501, 4893, replaceFileName2, 8494, 9004);
    QString placeToSave("output3.wav");
    reader.writeWav(placeToSave);
    */

    /*
    // test
    QString originalFileName = "data/test1/Videos/test1.wav";
    QString replaceFileName1 = "data/test1/Ads/Subway_Ad_15s.wav";
    QString replaceFileName2 = "data/test1/Ads/Starbucks_Ad_15s.wav";
    WavReader reader(originalFileName);
    reader.processOriginalFile();
    reader.replaceWav(replaceFileName1, 3220, 3668, replaceFileName2, 7322, 7640);
    reader.removeWav(5727, 6177 + 15);
    QString placeToSave("output_test.wav");
    reader.writeWav(placeToSave);
    */



}