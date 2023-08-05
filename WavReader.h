#ifndef WAVREADER_H
#define WAVREADER_H
#include <QApplication>
#include <QtEndian>
#include <QDebug>
#include <QFile>
#include <QString>
#include <vector>
using std::vector;

//Wav Header
struct wav_header_t
{
    // The 'RIFF' chunk
    char chunkID[4];
    quint32 chunkSize;
    char format[4]; //"WAVE"
    // The 'fmt' sub-chunk
    char subchunk1ID[4];
    quint32 subchunk1Size;
    quint16 audioFormat;
    quint16 numChannels; // 1
    quint32 sampleRate; // 48000 HZ
    quint32 byteRate;
    quint16 blockAlign;
    quint16 bitsPerSample; // 16
    // The 'data' sub-chunk
    char dataID[4];
    quint32 dataSize;
};

class WavReader
{
public:

    WavReader() {};

    WavReader(const QString filename): originalFilename(filename) { }

    QString getOriginalFilename() { return originalFilename; }

    void processOriginalFile();

    wav_header_t getWavHeader() { return wavHeader; }

    void setWavHeader(const wav_header_t header) { wavHeader = header; }

    vector<qint16> getData() { return originalData; }

    void setData(vector<qint16> data) { originalData = data; }

    void setOriginalFilename(const QString filename) { originalFilename = filename; }

    void replaceWav(const QString, int, int, const QString, int, int);

    void removeWav(int, int);

    void writeWav(const QString);

    ~WavReader() {};

private:

    QString originalFilename;

    void processWav(const QString, wav_header_t&, vector<qint16>&);

    wav_header_t wavHeader;

    // replace the sound of the longer ad in the original video's sound file
    void replaceLongerAd(const QString, int, int);
    // replace the sound of the shorter ad in the original video's sound file
    void replaceShorterAd(const QString, int, int, int, int);

    int diff;

    vector<qint16> originalData;
};
#endif //WAVREADER_H
