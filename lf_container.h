#ifndef LF_CONTAINER_H
#define LF_CONTAINER_H

#include <QFileInfo>
#include <QDir>

#include <hdf5.h>

#include <opencv2/opencv.hpp>

#include <iostream>

class LfContainer {

public:
    enum Dataset {
        HEIDELBERG,
        STANFORD
    };
    LfContainer(LfContainer::Dataset type);
    virtual ~LfContainer();
    virtual void getImage(uint16_t s, uint16_t t, cv::Mat &img) = 0;
    virtual void getEPIVT(uint16_t v, uint16_t t, cv::Mat &epi) = 0;
    virtual void getEPIUS(uint16_t u, uint16_t s, cv::Mat &epi) = 0;
    virtual void getGTDepth(uint16_t s, uint16_t t, cv::Mat &depth) = 0;
    virtual void convertDepthToDisparity(const cv::Mat& depth, cv::Mat &disp) = 0;
    virtual void convertDisparityToDepth(const cv::Mat& disp, cv::Mat &depth) = 0;
    uint16_t v();
    uint16_t u();
    uint16_t t();
    uint16_t s();

private:
    LfContainer::Dataset type;

protected:
    uint16_t _v;
    uint16_t _u;
    uint16_t _t;
    uint16_t _s;
    uint8_t _channels;
};

class HDF5Container: public LfContainer{
public:
    HDF5Container(QString path);
    ~HDF5Container();
    virtual void getImage(uint16_t s, uint16_t t, cv::Mat &img);
    virtual void getEPIVT(uint16_t v, uint16_t t, cv::Mat &epi);
    virtual void getEPIUS(uint16_t u, uint16_t s, cv::Mat &epi);
    virtual void getGTDepth(uint16_t s, uint16_t t, cv::Mat &depth);
    virtual void convertDepthToDisparity(const cv::Mat& depth, cv::Mat &disp);
    virtual void convertDisparityToDepth(const cv::Mat& disp, cv::Mat &depth);

private:
    hid_t _file, _group;
    float _dH, _focalLength, _shift;
};

class JPEGContainer: public LfContainer {

private:
    QFileInfo _path;
    std::vector<QString> _images;

public:
    JPEGContainer(QString path);
    ~JPEGContainer();

    virtual void getImage(uint16_t s, uint16_t t, cv::Mat &img);
    virtual void getEPIVT(uint16_t v, uint16_t t, cv::Mat &epi);
    virtual void getEPIUS(uint16_t u, uint16_t s, cv::Mat &epi);
    virtual void getGTDepth(uint16_t s, uint16_t t, cv::Mat &depth);
    virtual void convertDepthToDisparity(const cv::Mat& depth, cv::Mat &disp);
    virtual void convertDisparityToDepth(const cv::Mat& disp, cv::Mat &depth);

};

#endif // LF_CONTAINER_H
