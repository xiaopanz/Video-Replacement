#include "WavReader.h"

void WavReader::processWav(const QString filename, wav_header_t & wavHeader, vector<qint16>& data) {
    QFile wavFile(filename);
    if (!wavFile.open(QIODevice::ReadOnly))
    {
        qDebug() << "Error: Could not open file!";
        return;
    }
    //Read WAV file header
    QDataStream analyzeHeader (&wavFile);
    analyzeHeader.setByteOrder(QDataStream::LittleEndian);
    analyzeHeader.readRawData(wavHeader.chunkID, 4); // "RIFF"
    analyzeHeader >> wavHeader.chunkSize; // File Size
    analyzeHeader.readRawData(wavHeader.format,4); // "WAVE"
    analyzeHeader.readRawData(wavHeader.subchunk1ID,4); // "fmt"
    analyzeHeader >> wavHeader.subchunk1Size; // Format length
    analyzeHeader >> wavHeader.audioFormat; // Format type
    analyzeHeader >> wavHeader.numChannels; // Number of channels
    analyzeHeader >> wavHeader.sampleRate; // Sample rate
    analyzeHeader >> wavHeader.byteRate; // (Sample Rate * BitsPerSample * Channels) / 8
    analyzeHeader >> wavHeader.blockAlign; // (BitsPerSample * Channels) / 8.1
    analyzeHeader >> wavHeader.bitsPerSample; // Bits per sample
    char dataString[5] = "data";
    memcpy(wavHeader.dataID, dataString, 4);
    //Print WAV header
//    qDebug() << "WAV File Header read:";
//    qDebug() << "File Type: " << wavHeader.chunkID;
//    qDebug() << "File Size: " << wavHeader.chunkSize;
//    qDebug() << "WAV Marker: " << wavHeader.format;
//    qDebug() << "Format Name: " << wavHeader.subchunk1ID;
//    qDebug() << "Format Length: " << wavHeader.subchunk1Size;
//    qDebug() << "Format Type: " << wavHeader.audioFormat;
//    qDebug() << "Number of Channels: " << wavHeader.numChannels;
//    qDebug() << "Sample Rate: " <<  wavHeader.sampleRate;
//    qDebug() << "Sample Rate * Bits/Sample * Channels / 8: " << wavHeader.byteRate;
//    qDebug() << "Bits per Sample * Channels / 8.1: " << wavHeader.blockAlign;
//    qDebug() << "Bits per Sample: " << wavHeader.bitsPerSample;

    //Search data chunk
    quint32 chunkDataSize = 0;
    QByteArray temp_buff;
    char buff[4];
    while (true)
    {
        QByteArray tmp = wavFile.read(4);
        temp_buff.append(tmp);
        int idx = temp_buff.indexOf("data");
        if (idx >= 0)
        {
            int lenOfData = temp_buff.length() - (idx + 4);
            memcpy(buff, temp_buff.constData() + idx + 4, lenOfData);
            int bytesToRead = 4 - lenOfData;
            // finish reading size of chunk
            if (bytesToRead > 0)
            {
                int read = wavFile.read(buff + lenOfData, bytesToRead);
                if (bytesToRead != read)
                {
                    qDebug() << "Error: not reading enough bytes!";
                    return;
                }
            }
            chunkDataSize = qFromLittleEndian<quint32>((const uchar*)buff);
            wavHeader.dataSize = chunkDataSize;
            qDebug() << "Data size: " << wavHeader.dataSize;
            break;
        }
        if (temp_buff.length() >= 8)
        {
            temp_buff.remove(0, 0x04);
        }
    }
    if (!chunkDataSize)
    {
        qDebug() << "Error: Chunk data not found!";
        return;
    }


    //Reading data from the file
    int samples = 0;

    data.resize(chunkDataSize/2, 0);
    while (wavFile.read(buff, 4) > 0)
    {
        chunkDataSize -= 4;

        qint16 sampleChannel1 = qFromLittleEndian<qint16>((const uchar*)buff);
        qint16 sampleChannel2 = qFromLittleEndian<qint16>((const uchar*)(buff + 2));
        data[2 * samples] = sampleChannel1;
        data[2 * samples + 1] = sampleChannel2;
        ++samples;
        // check the end of the file
        if (chunkDataSize == 0 || chunkDataSize & 0x80000000)
            break;
    }
    qDebug() << "Size of data vector: " << data.size();
    qDebug() << "Finish Processing Wav. Read " << samples << " samples.";

    wavFile.close();
    return ;
}

void WavReader::processOriginalFile() {
    wav_header_t header;
    vector<qint16> data;
    WavReader::processWav(originalFilename, header, data);
    originalData = data;
    wavHeader = header;
}

//void WavReader::replaceWav(const QString filename, int pos) {
//    wav_header_t replaceHeader;
//    vector<qint16> replaceData;
//    WavReader::processWav(filename, replaceHeader, replaceData);
//    for (int i = pos; i < pos + replaceData.size(); ++ i)
//        originalData[i] = replaceData[i];
//}

void WavReader::replaceLongerAd(const QString filename, int start, int end) {
    wav_header_t replaceHeader;
    vector<qint16> replaceData;
    WavReader::processWav(filename, replaceHeader, replaceData);
    start = start * 1600;
    end = end * 1600;
    qDebug() << "start:" << start << end << replaceData.size();
    if (end - start >= replaceData.size()) {
        diff = end - start - replaceData.size();
        qDebug() << "diff:" << diff;
        for (int i = start; i < start + replaceData.size(); ++ i)
            originalData[i] = replaceData[i - start];
        for (int i = start + replaceData.size(); i < originalData.size() - diff; ++i)
            originalData[i] = originalData[i + diff];

    } else {
        diff = 0;
        for (int i = start; i < start + replaceData.size(); ++ i)
            originalData[i] = replaceData[i - start];
    }
    //qDebug() << "The original ad in the video is" << (float) diff / 48000 << "s shorter than 15s";
}

void WavReader::replaceShorterAd(const QString filename, int start, int end, int longerStart, int longerEnd) {
    wav_header_t replaceHeader;
    vector<qint16> replaceData;
    WavReader::processWav(filename, replaceHeader, replaceData);
    start = start * 1600;
    end = end * 1600;
    //qDebug() << "The start time is " << start / 48000 << "s";

    if (longerEnd * 1600 < start) {
        if (diff > 0) {
            for (int i = originalData.size() - 1; i >= start - diff + replaceData.size(); --i)
                originalData[i] = originalData[i - diff];
        }
        for (int i = start - diff; i < start - diff + replaceData.size(); ++i)
            originalData[i] = replaceData[i - start + diff];

    } else if (end < longerStart * 1600) {
        if (diff > 0) {
            for (int i = originalData.size() - 1 ; i >= end + diff; --i)
                originalData[i] = originalData[i - diff];
        }
        for (int i = start; i < start + replaceData.size(); ++i)
            originalData[i] = replaceData[i - start];
    } else {
        qDebug() << "Error! Two timestamps have overlap.";
    }
}

void WavReader::removeWav(int start, int end) {
    int rm_length = (end - start) * 1600;
    for (int i = start * 1600; i < originalData.size() - rm_length; ++i)
        originalData[i] = originalData[i + rm_length];
    for (int i = originalData.size() - rm_length; i <= originalData.size(); ++i)
        originalData[i] = 0;
}

void WavReader::replaceWav(const QString wav1, int start1, int end1, const QString wav2 , int start2, int end2) {
    if (end1 - start1 > end2 - start2) {
        replaceLongerAd(wav1, start1, end1);
        replaceShorterAd(wav2, start2, end2, start1, end1);
    } else {
        replaceLongerAd(wav2, start2, end2);
        replaceShorterAd(wav1, start1, end1, start2, end2);
    }


}

void WavReader::writeWav(const QString filename) {
    QFile origin(originalFilename);
    QFile saveFile(filename);
    saveFile.open(QIODevice::WriteOnly);
    if (!origin.open(QIODevice::ReadOnly)) {
        qDebug() << "Error: Could not open file!";
        return;
    }
    char buff[4];
    for(int i = 0; i < 19;++i){
        origin.read(buff, 4);
        saveFile.write(buff, 4);
    }
    origin.read(buff, 2);
    saveFile.write(buff, 2);

    origin.close();
    char* body = new char[wavHeader.dataSize];
    for(int i =0; i < originalData.size(); ++i) {
        short src = static_cast<short>(originalData[i]);
       body[2 * i] = src;
       src = src >> 8;
       body[2 * i +1] = src;
    }
    saveFile.write(body, wavHeader.dataSize);
    delete[] body;
    saveFile.close();
    return;
}