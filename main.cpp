/**
 * File              : src/%{Cpp:License:FileName}
 * Author            : Siddharth J. Singh <j.singh.logan@gmail.com>
 * Date              : 10.08.2017
 * Last Modified Date: 1.11.2018
 * Last Modified By  : Siddharth J. Singh <j.singh.logan@gmail.com>
 */

/**
 * src/%{Cpp:License:FileName}
 * Copyright (c) 2017 Siddharth J. Singh <j.singh.logan@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cmath>
#include <iostream>
#include <QString>
#include "lf_container.h"
#include "ledepthestimator.h"
//#include "Evaluator.h"
//#include "FastGCDepth.h"
//#include "Proposer.h"
//#include "StereoEnergy.h"

using namespace std;



//Depth estimator code

int main(int argc, char** argv)
{    
    LfContainer *light_field;
    QString lf_location;
    if(argc > 1)
        lf_location = argv[1];
    else
        lf_location = "/home/siddharth/workspace/lf_datasets/heidelberg/buddha/lf.h5";

    light_field = new HDF5Container(lf_location);

    std::string save_dir;
    if(argc > 1)
        save_dir = "output/" + std::string(argv[2]);
    else
        save_dir =  "output/debug/";

    #pragma omp parallel
    {
        //Initilize RNG with a different seed so that we get different
        //random planes everytime
        time_t seed;
        time(&seed);
        cv::setRNGSeed(uint64_t(seed));
    }

//    cv::Mat image;
//    light_field->getImage(4, 4, image);
//    cv::Mat image_g;
//    cv::cvtColor(image, image_g, cv::COLOR_BGR2GRAY);
//    cv::imwrite("lf_img.png", image_g);

    LEDepthEstimator *depth_estimator;
    depth_estimator = new LEDepthEstimator(light_field, -1.9, 1.9, save_dir);
//    depth_estimator->tests();
    depth_estimator->run();

    //Utility functions
//    cv::Mat cv_disp_img;
//    depth_estimator->computeDisparityFromPlane(cv_disp_img);
//    cv::Mat cv_depth_img;
//    light_field->convertDisparityToDepth(cv_disp_img, cv_depth_img);



    //write_epi_with_grad(horiz_epi, horiz_epi_xgrad, horiz_epi_ygrad);

    //render(gt_depth.row(row), cv_depth_img.row(row));

//    cv::Mat cv_depth_img;
//    light_field->convertDisparityToDepth(cv_disp_img, cv_depth_img);
//    depth_estimator->setCurrentDepth(cv_depth_img);
//    depth_estimator->setCurrentDisparity(cv_disp_img);

//    std::cout<<depth_estimator->getMSEdepth()<<std::endl;
//    std::cout<<depth_estimator->getMSEdisparity()<<std::endl;

    delete light_field;
    delete depth_estimator;

    return 0;
}

/* GC test
int main(int argc, char** argv)
{
    LfContainer *light_field;
    QString lf_location;
    if(argc > 1)
        lf_location = argv[1];
    else
        lf_location = "/home/siddharth/workspace/lf_datasets/heidelberg/buddha/lf.h5";

    light_field = new HDF5Container(lf_location);

    cv::Mat disp_gt, depth_gt;
    light_field->getGTDepth(light_field->s()/2, light_field->t()/2, depth_gt);

    #pragma omp parallel
    {
        //Initilize RNG with a different seed so that we get different
        //random planes everytime
        time_t seed;
        time(&seed);
        cv::setRNGSeed(uint64_t(seed));
    }

    Parameters param;
//    param.lambda = 1.0f;

    string out_dir;
    if(argc > 1)
        out_dir = param.output_dir + argv[2];
    else
        out_dir = param.output_dir + "/debug";

    Evaluator *eval = new Evaluator(depth_gt, "result", out_dir);
//    eval->setPrecision(calib.gt_prec);
    eval->showProgress = false;
//    eval->setErrorThreshold(errorThresh);
    Parameters params;

    FastGCDepth estimator(light_field, params);//, para, -1.9f);
    estimator.setEvaluator(eval);

    IProposer *prop1 = new ExpansionProposer(1);
    IProposer *prop2 = new RandomProposer(7, params.max_disp, params.min_disp);
    IProposer *prop3 = new ExpansionProposer(2);
    IProposer *prop4 = new RansacProposer(1);

    estimator.addLayer(5, {prop1, prop4, prop2});
    estimator.addLayer(15, {prop3, prop4});
//    estimator.addLayer(7, {prop3, prop4});

    cv::Mat labeling, disp;
    estimator.run(5, labeling, disp, {HORIZ}, 2);

//    Parameters params;
//    NaiveStereoEnergy nse(light_field, params, 1.9, -1.9, false);

//    Plane plane(0 , 0, -0.65f);
//    std::cout<<plane.GetZ(0 , 0)<<std::endl;

//    cv::Mat cost = cv::Mat::zeros(768, 768, CV_32F);
//    cv::Rect unit(20, 20, 2, 2);

//    nse.ComputeUnaryPotential(unit, cost, plane, HORIZ);

//    std::cout<<"Testing costs: "<<std::endl;

//    for(int i = unit.y; i < unit.y + unit.height; i++){
//        for(int j = unit.x; j < unit.y + unit.width; j++) {
//            std::cout<<cost.at<float>(i, j)<<", ";
//        }
//        std::cout<<std::endl;
//    }



//    cv::Mat m = (cv::Mat_<uchar>(3,2) << 1,2,3,4,5,6);

//    cv::Mat col_sum, row_sum;

//    std::cout<<"running :"<<m.type()<<std::endl;

//    cv::reduce(m, col_sum, 0, cv::REDUCE_SUM, CV_32F);
//    cv::reduce(m, row_sum, 1, cv::REDUCE_SUM, CV_32F);

//    for(int i = 0; i < 3; i++) {
//        std::cout<<col_sum.at<float>(0, i)<<", "<<std::endl;
//    }

//    std::cout<<std::endl;

//    cv::Mat epi_horiz;
//    light_field->getEPIVT(340, 4, epi_horiz);
//    cv::Mat channels[3];
//    cv::split(epi_horiz, channels);
//    for(int i = 0; i < 3; i++)
//        cv::equalizeHist(channels[i], channels[i]);

//    cv::Mat epi_horiz_eq;
//    cv::merge(channels, 3, epi_horiz_eq);
//    cv::imwrite("epi_horiz_eq.png", epi_horiz_eq);

////    cv::Mat epi_horiz_f;
//    epi_horiz_eq.convertTo(epi_horiz_eq, CV_32FC3);

//    cv::Mat epi_horiz_grad;
//    cv::Sobel(epi_horiz_eq, epi_horiz_grad, CV_32F, 1, 0, 1, 0.5, 0, cv::BORDER_REPLICATE);
//    cv::imwrite("epi_horiz_grad.png", epi_horiz_grad);

//    cv::Mat epi_middle(epi_horiz.rows, epi_horiz.cols + 4, epi_horiz.type(), cv::Scalar(0, 0, 0, 255));
//    cv::Rect epi_roi(2, 0, epi_horiz.cols, epi_horiz.rows);
//    epi_horiz.copyTo(epi_middle(epi_roi));// = epi_horiz.clone();
//    int mid_index = 4;
//    for(int i = 0; i < 9; i++) {
//        if(i != mid_index)
//            epi_middle.row(mid_index).copyTo(epi_middle.row(i));
//    }

//    cv::Mat epi_middle_f;

//    epi_middle.convertTo(epi_middle_f, CV_32FC3);

//    std::cout<<computeCostFromExtendedSet(-0.25f, epi_horiz, epi_middle_f, cv::Point2f(0, 0))<<std::endl;

}*/

//VCells test code
/*int main() {

    cv::Mat image = cv::imread("lf_img.png");
    VCells vc(100, 5.0);

    vc.bmpWidth = image.cols;
    vc.bmpHeight = image.rows;
    vc.lineByte = vc.bmpWidth;

    struct pixel* pixelArray = new pixel[vc.bmpHeight*vc.bmpWidth];

    for (int i = 0; i < vc.bmpHeight; i++) {
        for (int j = 0; j < vc.bmpWidth; j++) {
            int index = vc.getIndexFromRC(i, j);
            pixelArray[index].color[0] = image.at<cv::Vec3b>(i, j)[0];
            pixelArray[index].color[1] = image.at<cv::Vec3b>(i, j)[1];
            pixelArray[index].color[2] = image.at<cv::Vec3b>(i, j)[2];
        }
    }

    struct centroid* generators = new centroid[100];
    vc.initializePixel(pixelArray);
    vc.initializeGenerators(generators, pixelArray);
    vc.classicCVT(pixelArray, generators);
    //vc.EWCVT(generators, pixelArray);

    cv::Mat boundary;
    boundary = cv::Mat(image.rows, image.cols, CV_8U, cv::Scalar(255));

    for (int i = 0; i < vc.bmpHeight; i++) {
        for (int j = 0; j < vc.bmpWidth; j++) {
            int index = vc.getIndexFromRC(i, j);
            if(vc.isBoundaryPixel(&pixelArray[index], pixelArray))
                boundary.at<int>(i, j) = 0;
        }
    }

    cv::imwrite("labels.png", boundary);

}*/

/*
#define EXT_SET_SIZE 2

double computeCostFromExtendedSet(float disparity, cv::Mat epi, cv::Mat epi_middle, cv::Point2f center) {

    CV_Assert(disparity != 0.f);

    cv::Rect img_domain(0, 0, 768, 9);

    cv::Mat epi_f;
    epi.convertTo(epi_f, CV_32FC3);
    cv::imwrite("win.png", epi);

    cv::Mat warp_epi = cv::Mat::zeros(epi_f.size(), epi_f.type());
    cv::Point2f src[3], dst[3];

    int factor = epi.rows/2;
    float length = epi.rows - 1;
    float shift = disparity * factor;
    cv::Size out_size;
    float width = 0.f;
    cv::Point2i middle_shift(-2, 0);
    cv::Rect region(center.x - EXT_SET_SIZE, 0, 2*EXT_SET_SIZE + 1, epi.rows);

    if(shift < 0.f) {
        src[0] = cv::Point2f(shift + center.x, 0.f);
        src[1] = cv::Point2f(shift + center.x , length);
        src[2] = cv::Point2f(-shift + center.x, 0.f);

        width = std::max(-2.f*shift, float(EXT_SET_SIZE));

        dst[0] = cv::Point2f(0.f, 0.f);
        dst[1] = cv::Point2f(width, length);
        dst[2] = cv::Point2f(width, 0.f);

    } else {
        src[0] = cv::Point2f(center.x - shift, 0.f);
        src[1] = cv::Point2f(center.x - shift, length);
        src[2] = cv::Point2f(center.x + shift, length);

        width = std::max(2.f*shift, float(EXT_SET_SIZE));

        dst[0] = cv::Point2f(width, 0.f);
        dst[1] = cv::Point2f(0.f, length);
        dst[2] = cv::Point2f(width, length);

    }

    out_size = cv::Size(2*round(width) + 1, epi.rows);

    cv::Mat warp_aff = cv::getAffineTransform(src, dst);

    cv::warpAffine(epi_f, warp_epi, warp_aff, out_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    cv::Mat mask = warp_epi > 0.f;
    std::cout<<"mask type: "<<mask.type()<<std::endl;
    {
        cv::Mat t_mask;
        mask.convertTo(t_mask, CV_32F, 1/255.);
        mask = t_mask;
    }

    if(shift < 0.f) {
        if(width > EXT_SET_SIZE) {
            cv::Rect sub_region(round(width) - EXT_SET_SIZE, 0, 2*EXT_SET_SIZE + 1, epi.rows);
            warp_epi = warp_epi(sub_region);
            mask = mask(sub_region);
        }

        cv::Rect out = region & img_domain;

        if(out != region) {

            cv::Rect trim_region(0, 0, region.width - out.width, region.height);
            if(region.x != out.x) {
                trim_region.x = out.x - region.width + out.width;
            } else {
                trim_region.x = out.x + out.width;
            }
            trim_region.x -= region.x;

            mask(trim_region).setTo(0);

        }

    } else {
        if(width > EXT_SET_SIZE) {
            cv::Rect sub_region(round(width) - EXT_SET_SIZE, 0, 2*EXT_SET_SIZE + 1, epi.rows);
            warp_epi = warp_epi(sub_region);
            mask = mask(sub_region);
        }

        cv::Rect out = region & img_domain;

        if(out != region) {

            cv::Rect trim_region(0, 0, region.width - out.width, region.height);
            if(region.x != out.x) {
                trim_region.x = out.x - region.width + out.width;
            } else {
                trim_region.x = out.x + out.width;
            }
            trim_region.x -= region.x;

            mask(trim_region).setTo(0);

        }
    }

    cv::Mat middle(warp_epi.size(), warp_epi.type());
    middle = epi_middle(region - middle_shift).clone();
//    cv::Mat mask_middle;
//    middle.copyTo(warp_middle, mask);

//    std::cout<<middle.type()<<"::"<<warp_epi.type()<<std::endl;

    cv::Mat diff = (middle - warp_epi);
    diff = diff.mul(diff);
    diff = diff.mul(mask);

//    cv::Mat mask_diff;

//    diff.copyTo(mask_diff, mask);
//    cv::Mat sq_diff;
//    mask_diff = mask_diff.mul(diff);
//    std::cout<<middle.at<cv::Vec3f>(2, 3)<<std::endl;
//    std::cout<<warp_epi.at<cv::Vec3f>(2, 3)<<std::endl;
//    std::cout<<diff.at<cv::Vec3f>(2, 3)<<std::endl;

    cv::Mat diff_sum = cvutils::channelSum(diff);
    cv::sqrt(diff_sum, diff_sum);

//    for(int i = 0; i < 9; i++) {
//        for(int j = 0; j < 5; j++) {
//            std::cout<<diff_sum.at<float>(i, j)<<", ";
//        }
//        std::cout<<std::endl;
//    }

//    std::cout<<diff_sum.type()<<std::endl;
//    cv::Mat m = (cv::Mat_<float>(3,2) << 1,2,3,4,5,6);

//    cv::Mat col_sum, row_sum;

//    cv::reduce(m, col_sum, 0, cv::REDUCE_SUM, CV_32F);
//    cv::reduce(m, row_sum, 1, cv::REDUCE_SUM, CV_32F);

    cv::Mat col_sum;//(1, 5, diff_sum.type());
    cv::reduce(diff_sum, col_sum, 0, cv::REDUCE_SUM);
//    for(int i = 0; i < 2*EXT_SET_SIZE + 1; i++)
//        std::cout<<col_sum.at<float>(0, i)<<", ";

//    std::cout<<std::endl;

    float sigma_s = 5;
    float sigma_c = 4;

    cv::Mat exp_weights(col_sum.size(), col_sum.type());
    cv::Point2i o(2, 4);
    for(int i = -EXT_SET_SIZE; i <= EXT_SET_SIZE; i++) {
        cv::Vec3f col_diff = middle.at<cv::Vec3f>(o) - middle.at<cv::Vec3f>(o + cv::Point(i, 0));
        float col_intensity = std::sqrt(col_diff.dot(col_diff));
//        std::cout<<i<<", "<<col_intensity<<", ";

        exp_weights.at<float>(0, i + EXT_SET_SIZE) = std::exp(-(std::abs(i)/sigma_s) - col_intensity/sigma_c);
//        std::cout<<exp_weights.at<float>(0, i + EXT_SET_SIZE)<<std::endl;
    }

//    diff.copyTo(mask_diff, mask);
//    mask_diff.mul(mask_diff);
//    std::cout<<mask_diff.at<cv::Vec3f>(2, 3)<<std::endl;
//    cv::sqrt(mask_diff, mask_diff)

    cv::imwrite("diff.png", diff);
    cv::imwrite("diff_sum.png", diff_sum);
    cv::imwrite("middle.png", middle);
    cv::imwrite("warp.png", warp_epi);
    cv::imwrite("mask.png", mask);

    return exp_weights.dot(col_sum);
}*/

