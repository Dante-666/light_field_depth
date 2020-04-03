#include "lf_container.h"

LfContainer::LfContainer(LfContainer::Dataset type) {
    this->_type = type;
    this->_s = 0;
    this->_t = 0;
    this->_u = 0;
    this->_v = 0;
    this->_channels = 0;
}

uint16_t LfContainer::v() {
    return this->_v;
}

uint16_t LfContainer::u() {
    return this->_u;
}

uint16_t LfContainer::t() {
    return this->_t;
}

uint16_t LfContainer::s() {
    return this->_s;
}

LfContainer::Dataset LfContainer::type()
{
   return this->_type;
}

LfContainer::~LfContainer() {}

HDF5Container::HDF5Container(QString path) : LfContainer(LfContainer::Dataset::HEIDELBERG) {
    this->_file = H5Fopen(path.toLatin1().data(), H5F_ACC_RDONLY, H5P_DEFAULT);
    this->_group = H5Gopen1(_file, "/");
    herr_t status;

    hid_t att_yRes = H5Aopen(this->_group, "yRes", H5P_DEFAULT);
    if((status = H5Aread(att_yRes, H5T_STD_I64LE, &this->_v)) < 0) {
        std::cerr<<"Failed to open attribute \"yRes\""<<std::endl;
        throw std::runtime_error("LF file improper");
    }
    if((status = H5Aclose(att_yRes)) < 0) {
        std::cerr<<"Failed to close attribute \"yRes\""<<std::endl;
        throw std::runtime_error("LF file improper");
    }

    hid_t att_xRes = H5Aopen(this->_group, "xRes", H5P_DEFAULT);
    if((status = H5Aread(att_xRes, H5T_STD_I64LE, &this->_u)) < 0) {
        std::cerr<<"Failed to open attribute \"xRes\""<<std::endl;
        throw std::runtime_error("LF file improper");
    }
    if((status = H5Aclose(att_xRes)) < 0) {
        std::cerr<<"Failed to close attribute \"xRes\""<<std::endl;
        throw std::runtime_error("LF file improper");
    }

    hid_t att_hRes = H5Aopen(this->_group, "hRes", H5P_DEFAULT);
    if((status = H5Aread(att_hRes, H5T_STD_I64LE, &this->_t)) < 0) {
        std::cerr<<"Failed to open attribute \"hRes\""<<std::endl;
        throw std::runtime_error("LF file improper");
    }
    if((status = H5Aclose(att_hRes)) < 0) {
        std::cerr<<"Failed to close attribute \"hRes\""<<std::endl;
        throw std::runtime_error("LF file improper");
    }

    hid_t att_vRes = H5Aopen(this->_group, "vRes", H5P_DEFAULT);
    if((status = H5Aread(att_vRes, H5T_STD_I64LE, &this->_s)) < 0) {
        std::cerr<<"Failed to open attribute \"vRes\""<<std::endl;
        throw std::runtime_error("LF file improper");
    }
    if((status = H5Aclose(att_vRes)) < 0) {
        std::cerr<<"Failed to close attribute \"vRes\""<<std::endl;
        throw std::runtime_error("LF file improper");
    }

    hid_t att_channels = H5Aopen(this->_group, "channels", H5P_DEFAULT);
    if((status = H5Aread(att_channels, H5T_STD_I64LE, &this->_channels)) < 0) {
        std::cerr<<"Failed to open attribute \"channels\""<<std::endl;
        throw std::runtime_error("LF file improper");
    }
    if((status = H5Aclose(att_channels)) < 0) {
        std::cerr<<"Failed to close attribute \"channels\""<<std::endl;
        throw std::runtime_error("LF file improper");
    }

    hid_t att_dH = H5Aopen(this->_group, "dH", H5P_DEFAULT);
    if((status = H5Aread(att_dH, H5T_IEEE_F32LE, &this->_dH)) < 0) {
        std::cerr<<"Failed to open attribute \"dH\""<<std::endl;
        throw std::runtime_error("LF file improper");
    }
    if((status = H5Aclose(att_dH)) < 0) {
        std::cerr<<"Failed to close attribute \"dH\""<<std::endl;
        throw std::runtime_error("LF file improper");
    }

    hid_t att_focalLength = H5Aopen(this->_group, "focalLength", H5P_DEFAULT);
    if((status = H5Aread(att_focalLength, H5T_IEEE_F32LE, &this->_focalLength)) < 0) {
        std::cerr<<"Failed to open attribute \"focalLength\""<<std::endl;
        throw std::runtime_error("LF file improper");
    }
    if((status = H5Aclose(att_focalLength)) < 0) {
        std::cerr<<"Failed to close attribute \"focalLength\""<<std::endl;
        throw std::runtime_error("LF file improper");
    }

    hid_t att_shift = H5Aopen(this->_group, "shift", H5P_DEFAULT);
    if((status = H5Aread(att_shift, H5T_IEEE_F32LE, &this->_shift)) < 0) {
        std::cerr<<"Failed to open attribute \"shift\""<<std::endl;
        throw std::runtime_error("LF file improper");
    }
    if((status = H5Aclose(att_shift)) < 0) {
        std::cerr<<"Failed to close attribute \"shift\""<<std::endl;
        throw std::runtime_error("LF file improper");
    }

}

HDF5Container::~HDF5Container() {
    herr_t status;

    if((status = H5Gclose(this->_group)) < 0) {
        std::cerr<<"Failed to close group"<<std::endl;
    }
    if((status = H5Fclose(this->_file)) < 0) {
        std::cerr<<"Failed to close file"<<std::endl;
    }
}

void HDF5Container::getImage(uint16_t s, uint16_t t, cv::Mat &img) {

    herr_t status;
    hid_t data_lf = H5Dopen2(this->_group, "LF", H5P_DEFAULT);

    hsize_t subset_size[5] = {1, 1, this->_v, this->_u, this->_channels};

    uint8_t data[this->_v][this->_u][this->_channels];

    hsize_t start[5] = {t, s, 0, 0, 0};
    hsize_t stride[5] = {1, 1, 1, 1, 1};
    hsize_t count[5] = {1, 1, 1, 1, 1};
    hsize_t block[5] = {1, 1, this->_v, this->_u, this->_channels};

    hid_t dataspace_lf = H5Dget_space(data_lf);

    if((status = H5Sselect_hyperslab(dataspace_lf, H5S_SELECT_SET, start, stride, count, block)) < 0)
        std::cerr<<"Unable to select the hyperslab."<<std::endl;

    hid_t memspace_lf = H5Screate_simple(5, subset_size, NULL);

    if((status = H5Dread(data_lf, H5T_STD_U8LE, memspace_lf, dataspace_lf, H5P_DEFAULT, data)) < 0)
        std::cerr<<"Unable to read the subset of data. "<<s<<", "<<t<<std::endl;

    img = cv::Mat(this->_v, this->_u, CV_8UC3);

    for(int i = 0 ; i < this->_v; i++) {
        for(int j = 0 ; j < this->_u; j++) {
            img.at<cv::Vec3b>(i, j)[0] = data[i][j][2];
            img.at<cv::Vec3b>(i, j)[1] = data[i][j][1];
            img.at<cv::Vec3b>(i, j)[2] = data[i][j][0];
        }
    }

    if((status = H5Sclose(memspace_lf)) < 0)
        std::cerr<<"Unable to close the allocated memory space."<<std::endl;
    if((status = H5Sclose(dataspace_lf)) < 0)
        std::cerr<<"Unable to close the dataspace."<<std::endl;
    if((status = H5Dclose(data_lf)) < 0)
        std::cerr<<"Unable to close dataset."<<std::endl;

}

void HDF5Container::getEPIVT(uint16_t v, uint16_t t, cv::Mat &epi) {

    herr_t status;
    hid_t data_lf = H5Dopen2(this->_group, "LF", H5P_DEFAULT);

    hsize_t subset_size[5] = {1, this->_s, 1, this->_u, this->_channels};

    uint8_t data[this->_s][this->_u][this->_channels];

    hsize_t start[5] = {t, 0, v, 0, 0};
    hsize_t stride[5] = {1, 1, 1, 1, 1};
    hsize_t count[5] = {1, this->_s, 1, 1, 1};
    hsize_t block[5] = {1, 1, 1, this->_u, this->_channels};

    hid_t dataspace_lf = H5Dget_space(data_lf);

    if((status = H5Sselect_hyperslab(dataspace_lf, H5S_SELECT_SET, start, stride, count, block)) < 0)
        std::cerr<<"Unable to select the hyperslab."<<std::endl;

    hid_t memspace_lf = H5Screate_simple(5, subset_size, NULL);

    if((status = H5Dread(data_lf, H5T_STD_U8LE, memspace_lf, dataspace_lf, H5P_DEFAULT, data)) < 0)
        std::cerr<<"Unable to read the subset of data."<<std::endl;

    epi = cv::Mat(this->_s, this->_u, CV_8UC3);

    for(int i = 0 ; i < this->_s; i++) {
        for(int j = 0 ; j < this->_u; j++) {
            epi.at<cv::Vec3b>(i, j)[0] = data[i][j][2];
            epi.at<cv::Vec3b>(i, j)[1] = data[i][j][1];
            epi.at<cv::Vec3b>(i, j)[2] = data[i][j][0];
        }
    }

    if((status = H5Sclose(memspace_lf)) < 0)
        std::cerr<<"Unable to close the allocated memory space."<<std::endl;
    if((status = H5Sclose(dataspace_lf)) < 0)
        std::cerr<<"Unable to close the dataspace."<<std::endl;
    if((status = H5Dclose(data_lf)) < 0)
        std::cerr<<"Unable to close dataset."<<std::endl;
}

void HDF5Container::getEPIUS(uint16_t u, uint16_t s, cv::Mat &epi) {

    herr_t status;
    hid_t data_lf = H5Dopen2(this->_group, "LF", H5P_DEFAULT);

    hsize_t subset_size[5] = {this->_t, 1, this->_v, 1, this->_channels};

    uint8_t data[this->_t][this->_v][this->_channels];

    hsize_t start[5] = {0, s, 0, u, 0};
    hsize_t stride[5] = {1, 1, 1, 1, 1};
    hsize_t count[5] = {this->_t, 1, 1, 1, 1};
    hsize_t block[5] = {1, 1, this->_v, 1, this->_channels};

    hid_t dataspace_lf = H5Dget_space(data_lf);

    if((status = H5Sselect_hyperslab(dataspace_lf, H5S_SELECT_SET, start, stride, count, block)) < 0)
        std::cerr<<"Unable to select the hyperslab."<<std::endl;

    hid_t memspace_lf = H5Screate_simple(5, subset_size, NULL);

    if((status = H5Dread(data_lf, H5T_STD_U8LE, memspace_lf, dataspace_lf, H5P_DEFAULT, data)) < 0)
        std::cerr<<"Unable to read the subset of data."<<std::endl;

    epi = cv::Mat(this->_v, this->_t, CV_8UC3);

    for(int i = 0 ; i < this->_v; i++) {
        for(int j = 0 ; j < this->_t; j++) {
            epi.at<cv::Vec3b>(i, j)[0] = data[j][i][2];
            epi.at<cv::Vec3b>(i, j)[1] = data[j][i][1];
            epi.at<cv::Vec3b>(i, j)[2] = data[j][i][0];
        }
    }

    if((status = H5Sclose(memspace_lf)) < 0)
        std::cerr<<"Unable to close the allocated memory space."<<std::endl;
    if((status = H5Sclose(dataspace_lf)) < 0)
        std::cerr<<"Unable to close the dataspace."<<std::endl;
    if((status = H5Dclose(data_lf)) < 0)
        std::cerr<<"Unable to close dataset."<<std::endl;

}

void HDF5Container::getGTDepth(uint16_t s, uint16_t t, cv::Mat &depth) {

    herr_t status;
    hid_t data_depth = H5Dopen2(this->_group, "GT_DEPTH", H5P_DEFAULT);

    hsize_t subset_size[4] = {1, 1, this->_v, this->_u};

    float data[this->_v][this->_u];

    hsize_t start[4] = {t, s, 0, 0};
    hsize_t stride[4] = {1, 1, 1, 1};
    hsize_t count[4] = {1, 1, 1, 1};
    hsize_t block[4] = {1, 1, this->_v, this->_u};

    hid_t dataspace_lf = H5Dget_space(data_depth);

    if((status = H5Sselect_hyperslab(dataspace_lf, H5S_SELECT_SET, start, stride, count, block)) < 0)
        std::cerr<<"Unable to select the hyperslab."<<std::endl;

    hid_t memspace_lf = H5Screate_simple(4, subset_size, NULL);

    if((status = H5Dread(data_depth, H5T_IEEE_F32LE, memspace_lf, dataspace_lf, H5P_DEFAULT, data)) < 0)
        std::cerr<<"Unable to read the subset of data. ";//<<s<<", "<<t<<std::endl;

    depth = cv::Mat(this->_v, this->_u, CV_32F);

    for(int i = 0 ; i < this->_v; i++) {
        for(int j = 0 ; j < this->_u; j++) {
            depth.at<float>(i, j) = data[i][j];
        }
    }

    if((status = H5Sclose(memspace_lf)) < 0)
        std::cerr<<"Unable to close the allocated memory space."<<std::endl;
    if((status = H5Sclose(dataspace_lf)) < 0)
        std::cerr<<"Unable to close the dataspace."<<std::endl;
    if((status = H5Dclose(data_depth)) < 0)
        std::cerr<<"Unable to close dataset."<<std::endl;

}

void HDF5Container::convertDepthToDisparity(const cv::Mat& depth, cv::Mat &disp) {
    disp = double(this->_dH * this->_focalLength)/depth - double(this->_shift);
}

void HDF5Container::convertDisparityToDepth(const cv::Mat& disp, cv::Mat &depth) {
    depth = double(this->_dH * this->_focalLength)/(disp + double(this->_shift));
}

JPEGContainer::JPEGContainer(QString path) : LfContainer(LfContainer::Dataset::STANFORD) {

    QStringList files = QDir(path).entryList(QDir::Filter::Files);

    for (int i = 0; i < files.size(); i++) {
        QFileInfo f(files[i]);
        if (f.suffix() == "png") {
            uint16_t s, t;
            float c_s, c_t;
            char ext[10];
            int ret = sscanf(files[i].toStdString().c_str(),
                             "out_%hu_%hu_%f_%f.%s", &t, &s, &c_t, &c_s, ext);

            if (ret == 5) {
                if(this->_t < t) this->_t = t;
                if(this->_s < s) this->_s = s;
                this->_images.emplace_back(path + '/' + files[i]);
            } else {
                std::cerr << "[ERROR] File name is invalid!!" << std::endl;
                std::cerr << "        It must be like \"out_%d_%d_-%f_%f.%s\"" << std::endl;
                std::abort();
            }
        }
        if(f.suffix() == "yml") {
            std::string calib_path = path.toStdString() + '/' + files[i].toStdString();
            cv::FileStorage calib(calib_path, cv::FileStorage::READ);
            calib["_dH"] >> _dH;
            calib["_focalLength"] >> _focalLength;
            calib["_shift"] >> _shift;

        }
    }


    this->_t++;
    this->_s++;

    cv::Mat img = cv::imread(this->_images[0].toStdString());

    this->_u = uint16_t(img.cols);
    this->_v = uint16_t(img.rows);
    this->_channels = uint8_t(img.channels());

}

JPEGContainer::~JPEGContainer() {
    this->_images.clear();
}

void JPEGContainer::getImage(uint16_t s, uint16_t t, cv::Mat &img) {
    uint index = t*this->_s + s;

    img = cv::imread(this->_images[index].toStdString());
}

void JPEGContainer::getEPIVT(uint16_t v, uint16_t t, cv::Mat &epi) {

    if(this->_channels == 3)
        epi = cv::Mat(this->_s, this->_u, CV_8UC3);
    else if(this->_channels == 1)
        epi = cv::Mat(this->_s, this->_u, CV_8UC1);

    for(uint i = 0; i < this->_s; i++) {
        uint index = t * this->_s + i;
        cv::Mat img = cv::imread(this->_images[index].toStdString());
        img.row(v).copyTo(epi.row(int(i)));
    }
}

void JPEGContainer::getEPIUS(uint16_t u, uint16_t s, cv::Mat &epi) {
    if(this->_channels == 3)
        epi = cv::Mat(this->_v, this->_t, CV_8UC3);
    else if(this->_channels == 1)
        epi = cv::Mat(this->_v, this->_t, CV_8UC1);

    for(uint i = 0; i < this->_t; i++) {
        uint index = i * this->_s + s;
        cv::Mat img = cv::imread(this->_images[index].toStdString());
        img.col(u).copyTo(epi.col(int(i)));
    }
}

void JPEGContainer::getGTDepth(uint16_t s, uint16_t t, cv::Mat &depth) {
    return;
}

void JPEGContainer::convertDepthToDisparity(const cv::Mat& depth, cv::Mat &disp) {
    disp = double(this->_dH * this->_focalLength)/depth - double(this->_shift);
}

void JPEGContainer::convertDisparityToDepth(const cv::Mat& disp, cv::Mat &depth) {
    depth = double(this->_dH * this->_focalLength)/(disp + double(this->_shift));
}
