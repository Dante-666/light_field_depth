/**
 * File              : src/ledepthestimator.h
 * Author            : Siddharth J. Singh <j.singh.logan@gmail.com>
 * Date              : 10.08.2017
 * Last Modified Date: 29.3.2019
 * Last Modified By  : Siddharth J. Singh <j.singh.logan@gmail.com>
 */

/**
 * src/ledepthestimator.h
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
#ifndef LEDEPTHESTIMATOR_H
#define LEDEPTHESTIMATOR_H

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
    const int unitRegionSize;

    int heightBlocks;
    int widthBlocks;

    std::vector<cv::Rect> unitRegions;
    std::vector<cv::Rect> sharedRegions;
    std::vector<std::vector<int>> disjointRegionSets;

    Region(int width, int height, int unitRegionSize) :
        height(height),
        width(width) ,
        unitRegionSize(unitRegionSize) {

        int minsize = std::max(2, unitRegionSize / 2);
        int frac_h = height % unitRegionSize;
        int frac_w = width % unitRegionSize;
        int split_h = frac_h >= minsize ? 1 : 0;
        int split_w = frac_w >= minsize ? 1 : 0;

        heightBlocks = (height / unitRegionSize) + split_h;
        widthBlocks = (width / unitRegionSize) + split_w;

        unitRegions.resize(heightBlocks * widthBlocks);
        sharedRegions.resize(heightBlocks * widthBlocks);
        disjointRegionSets.resize(16);

        cv::Rect imageDomain(0, 0, width, height);

        for (int i = 0; i < heightBlocks; i++) {
            for (int j = 0; j < widthBlocks; j++) {
                int r = i*widthBlocks + j;

                cv::Rect &unitRegion = unitRegions[r];
                cv::Rect &sharedRegion = sharedRegions[r];

                unitRegion.x = j * unitRegionSize;
                unitRegion.y = i * unitRegionSize;
                unitRegion.width = unitRegionSize;
                unitRegion.height = unitRegionSize;
                unitRegion = unitRegion & imageDomain;

                sharedRegion.x = (j - 1) * unitRegionSize;
                sharedRegion.y = (i - 1) * unitRegionSize;
                sharedRegion.width = unitRegionSize * 3;
                sharedRegion.height = unitRegionSize * 3;
                sharedRegion = sharedRegion & imageDomain;
            }
        }

        // Fix sizes of regions near left and bottom boundaries to include fractional regions.
        if (split_w == 0)
        {
            for (int i = 0; i < heightBlocks; i++) {
                int x1 = i * widthBlocks + widthBlocks - 1;
                unitRegions[x1].width += frac_w;
            }
            for (int i = 0; i < heightBlocks; i++) {
                int x1 = i * widthBlocks + widthBlocks - 2;
                sharedRegions[x1].width += frac_w;
            }
        }
        if (split_h == 0)
        {
            for (int j = 0; j < widthBlocks; j++) {
                int y1 = (heightBlocks - 1) * widthBlocks + j;
                unitRegions[y1].height += frac_h;
            }
            for (int j = 0; j < widthBlocks; j++) {
                int y1 = (heightBlocks - 2) * widthBlocks + j;
                sharedRegions[y1].height += frac_h;
            }
        }

        for ( int i = 0; i < heightBlocks; i++ ){
            for ( int j = 0; j < widthBlocks; j++ ){
                int r = i * widthBlocks + j;
                disjointRegionSets[(i%4)*4 + (j%4)].push_back(r);
            }
        }
        auto it = disjointRegionSets.begin();
        while (  it != disjointRegionSets.end() ){
            it->shrink_to_fit();
            if ( it->size() == 0 ){
                it = disjointRegionSets.erase(it);
            } else {
                it++;
            }
        }

    };
    ~Region() {};
};

class LEDepthEstimator
{
public:
    LEDepthEstimator(LfContainer *light_field, float min_disp, float max_disp, std::string save_dir);
    ~LEDepthEstimator();

    Plane createRandomLabel(cv::Point s) const;
    void initializeCoordinates();
    void initializeRandomPlane(enum epi_type type);
    void initializeSmoothnessCoeff();
    void initializeCurrentCostsFast(enum epi_type type);
    void initializeCurrentCosts(enum epi_type type);
    void computeDisparityFromPlane(enum epi_type type);
    float getDisparityPerturbationWidth(int iter);

    void tests();
    void run();

    bool isValidLabel(Plane label, cv::Point pos);
    cv::Mat isValidLabel(Plane label, cv::Rect rect);
    void prefetchEPIData(bool archived = false);

    void runHorizontalRegionPropagation(int iter, int grid, bool do_gc = false);
    void runVerticalRegionPropagation(int iter, int grid, bool do_gc = false);

    void runHorizontalIterativeExpansion(int iter, int grid, int set, bool do_gc = false);
    void runVerticalIterativeExpansion(int iter, int grid, int set, bool do_gc = false);

    void runHorizontalExpansionProposer(const Region& region, int set, bool do_gc = false);
    void runHorizontalRansacProposer(const Region& region, int set, bool do_gc = false);
    void runHorizontalRandomProposer(const Region& region, int iter, int set, bool do_gc = false);

    void runVerticalExpansionProposer(const Region& region, int set, bool do_gc = false);
    void runVerticalRansacProposer(const Region& region, int set, bool do_gc = false);
    void runVerticalRandomProposer(const Region& region, int iter, int set, bool do_gc = false);

    void runHorizontalCostComputation(cv::Mat& proposalCosts, cv::Mat& proposedLabels, cv::Mat& pixelMask);
    void runVerticalCostComputation(cv::Mat& proposalCosts, cv::Mat& proposedLabels, cv::Mat& pixelMask);

    void runGCExpansion(const cv::Rect& sharedRegion, const cv::Mat& localProposals, Plane alpha, cv::Mat& updateMask);

    Plane getExpansionPlane(const cv::Rect& unitRegion, epi_type type);
    Plane getRANSACPlane(const cv::Rect& unitRegion, epi_type type);
    Plane getRandomPlane(const cv::Rect& unitRegion, epi_type type, int iter);
    inline int computeSampleCount(int ni, int ptNum, int pf, double conf);
    inline bool isPixelInSet(cv::Point pt, const Region& region, int set);
    inline int getRegionIDfromPixel(cv::Point pt, const Region& region, int set);

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

//    double getMSEdepth();
//    double getMSEdisparity();

    void evaluate(int iter);

//    void render(const cv::Mat& depth, const cv::Mat& curr_estimate);
//    void write_epi_with_grad(const cv::Mat& epi, const cv::Mat& epi_xgrad, const cv::Mat& epi_ygrad, epi_type type);
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

#endif // LEDEPTHESTIMATOR_H
