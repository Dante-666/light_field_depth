/**
 * File              : src/depthestimator.h
 * Author            : Siddharth J. Singh <j.singh.logan@gmail.com>
 * Date              : 10.08.2017
 * Last Modified Date: 22.10.2018
 * Last Modified By  : Siddharth J. Singh <j.singh.logan@gmail.com>
 */

/**
 * src/depthestimator.h
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
#ifndef DEPTHESTIMATOR_H
#define DEPTHESTIMATOR_H

#include "lf_container.h"

#include <opencv2/ximgproc.hpp>

#include "ctime"
#include "Plane.h"
#include "maxflow/graph.h"
#include "Utilities.hpp"

#include <omp.h>
#include <chrono>
#include <ratio>
#include <algorithm>
#include "gnuplot-iostream.h"

enum epi_type { HORIZ, VERT, END };
enum neighbour_id {N, E, S, W, NE, SE, SW, NW};

class Region {
public:
    const int height;
    const int width;
    const int regionUnitSize;

    int heightBlocks;
    int widthBlocks;

    std::vector<cv::Rect> unitRegions;

    Region(int width, int height, int regionUnitSize) :
        height(height),
        width(width) ,
        regionUnitSize(regionUnitSize) {

        int minsize = std::max(2, regionUnitSize / 2);
        int frac_h = height % regionUnitSize;
        int frac_w = width % regionUnitSize;
        int split_h = frac_h >= minsize ? 1 : 0;
        int split_w = frac_w >= minsize ? 1 : 0;

        heightBlocks = (height / regionUnitSize) + split_h;
        widthBlocks = (width / regionUnitSize) + split_w;

        unitRegions.resize(heightBlocks * widthBlocks);
        cv::Rect imageDomain(0, 0, width, height);

        for (int i = 0; i < heightBlocks; i++) {
            for (int j = 0; j < widthBlocks; j++) {
                int r = i*widthBlocks + j;

                cv::Rect &unitRegion = unitRegions[r];

                unitRegion.x = j * regionUnitSize;
                unitRegion.y = i * regionUnitSize;
                unitRegion.width = regionUnitSize;
                unitRegion.height = regionUnitSize;
                unitRegion = unitRegion & imageDomain;
            }
        }

        // Fix sizes of regions near left and bottom boundaries to include fractional regions.
        if (split_w == 0)
        {
            for (int i = 0; i < heightBlocks; i++) {
                int x1 = i * widthBlocks + widthBlocks - 1;
                unitRegions[x1].width += frac_w;
            }
        }
        if (split_h == 0)
        {
            for (int j = 0; j < widthBlocks; j++) {
                int y1 = (heightBlocks - 1) * widthBlocks + j;
                unitRegions[y1].height += frac_h;
            }
        }

    };
    ~Region() {};
};

class PMDepthEstimator
{
public:
    PMDepthEstimator(LfContainer *light_field, float min_disp, float max_disp, std::string save_dir);
    ~PMDepthEstimator();

    Plane createRandomLabel(cv::Point s) const;
    void initializeCoordinates();
    void initializeRandomPlane(enum epi_type type);
    void initializeSmoothnessCoeff();
    void computeDisparityFromPlane(enum epi_type type);
    float getDisparityPerturbationWidth(int iter);

    void tests();
    void run();

    inline bool isValidLabel(Plane label, cv::Point pos);
    void prefetchEPIData(bool archive = false);
    void runHorizontalSpatialPropagation(int iter);
    void runVerticalSpatialPropagation(int iter);
    void perturbatePlaneLabels(int iter);

    void runHorizontalRegionPropagation(int iter);
    void runVerticalRegionPropagation(int iter);

    Plane getExpansionPlane(cv::Rect unitRegion, epi_type);
    Plane getRANSACPlane(cv::Rect unitRegion, epi_type type);
    Plane getRandomPlane(cv::Rect unitRegion, epi_type type, int iter);
    inline int computeSampleCount(int ni, int ptNum, int pf, double conf);
    inline int getRegionIDfromPixel(cv::Point pt, int region_id);

    double getCostFromExtendedSet(const cv::Point2f& centre,
                                  epi_type type,
                                  const float disparity,
                                  const cv::Mat &epi,
                                  const cv::Mat &epi_grad);

    inline void getPixelSet(const cv::Point2f& centre,
                            std::vector<cv::Point2f>& pixel_set,
                            epi_type type,
                            const float disparity);

    inline double getWeightedColorAndGradScore(const cv::Point2f& center,
                                               const std::vector<cv::Point2f> pixel_set,
                                               epi_type type,
                                               const cv::Mat& epi,
                                               const cv::Mat& epi_grad);

    void postProcess();

    double getMSEdepth();
    double getMSEdisparity();

    void evaluate(int iter);

    void render(const cv::Mat& depth, const cv::Mat& curr_estimate);
    void write_epi_with_grad(const cv::Mat& epi, const cv::Mat& epi_xgrad, const cv::Mat& epi_ygrad, epi_type type);
    void serializeEPIData();
    void deserializeEPIData();

private:
    LfContainer *light_field;

    std::vector<Region> regions;

    uint16_t width;
    uint16_t height;
    float min_disp;
    float max_disp;

    std::vector<cv::Mat> h_epi_arr;
    std::vector<cv::Mat> h_epi_grad_arr;

    std::vector<cv::Mat> v_epi_arr;
    std::vector<cv::Mat> v_epi_grad_arr;

    cv::Mat current_plane_label[2];
    cv::Mat current_disp[2];
    cv::Mat current_depth[2];
    cv::Mat coordinates;

    cv::Mat gt_disp;
    cv::Mat gt_depth;

    cv::Mat current_cost[2];

    std::vector<cv::Point> neighbours;
    std::vector<cv::Mat> smoothness_coeff;

    //Parameters
    float alpha = 0.8;
    int W = 2;
    int M = 1;
    double thresh_color = 200.f * (1 - alpha);
    double thresh_gradient = 57.f * alpha;
    double thresh_smooth = 1.0f;

    std::string save_dir;
    int INVALID_COST = 1000000;

};

#endif // DEPTHESTIMATOR_H
