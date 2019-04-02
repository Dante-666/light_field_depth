/**
 * File              : src/ledepthestimator.cpp
 * Author            : Siddharth J. Singh <j.singh.logan@gmail.com>
 * Date              : 10.08.2017
 * Last Modified Date: 29.3.2019
 * Last Modified By  : Siddharth J. Singh <j.singh.logan@gmail.com>
 */

/**
 * src/ledepthestimator.cpp
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
#include "ledepthestimator.h"

LEDepthEstimator::LEDepthEstimator(LfContainer *light_field, float min_disp, float max_disp, std::string save_dir)
{
    this->light_field = light_field;
    this->width = light_field->u();
    this->height = light_field->v();

    this->regions.push_back(Region(this->width-2*M, this->height-2*M, 5));
    this->regions.push_back(Region(this->width-2*M, this->height-2*M, 15));
    this->regions.push_back(Region(this->width-2*M, this->height-2*M, 25));

    this->min_disp = min_disp;
    this->max_disp = max_disp;
    this->save_dir = save_dir;

    current_plane_label[HORIZ] = cv::Mat::zeros(this->height, this->width, cv::DataType<Plane>::type);
    current_disp[HORIZ] = cv::Mat::zeros(this->height, this->width, CV_32F);
    current_plane_label[VERT] = cv::Mat::zeros(this->height, this->width, cv::DataType<Plane>::type);
    current_disp[VERT] = cv::Mat::zeros(this->height, this->width, CV_32F);
    coordinates = cv::Mat::zeros(this->height, this->width, cv::DataType<Plane>::type);

    current_cost[HORIZ] = cv::Mat::zeros(this->height, this->width, CV_32F);
    current_cost[VERT] = cv::Mat::zeros(this->height, this->width, CV_32F);


    neighbours.resize(8);
    neighbours[W] = cv::Point(-1, +0);
    neighbours[E] = cv::Point(+1, +0);
    neighbours[N] = cv::Point(+0, -1);
    neighbours[S] = cv::Point(+0, +1);

    neighbours[NW] = cv::Point(-1, -1);
    neighbours[NE] = cv::Point(+1, -1);
    neighbours[SW] = cv::Point(-1, +1);
    neighbours[SE] = cv::Point(+1, +1);

    //This generates a vector of coordinates that can be used for inner product
    this->initializeCoordinates();

    //One time effort for both EPIs
    this->initializeSmoothnessCoeff();

    this->prefetchEPIData();

    //This generates random planes and associates them with each pixel
//    this->initializeRandomPlane(HORIZ);
//    this->initializeRandomPlane(VERT);
    //Init current fast
//    this->initializeCurrentCosts(HORIZ);
//    this->initializeCurrentCosts(VERT);

    //This generates region wise proposals and is better at supressing invalid labeling
    this->initializeCurrentCostsFast(HORIZ);
    this->initializeCurrentCostsFast(VERT);

    //Set the ground truths for central image from the light field data
    this->light_field->getGTDepth(this->light_field->s()/2, this->light_field->t()/2, this->gt_depth);
    this->light_field->convertDepthToDisparity(this->gt_depth, this->gt_disp);
}

LEDepthEstimator::~LEDepthEstimator()
{

}

Plane LEDepthEstimator::createRandomLabel(cv::Point s) const
{
    float zs = cv::theRNG().uniform(float(min_disp), float(max_disp));
    float vs = 0.f;

    cv::Vec3d n = cvutils::getRandomUnitVector(CV_PI / 3);
    return Plane::CreatePlane(n, zs, float(s.x), float(s.y), vs);
}

void LEDepthEstimator::initializeCoordinates()
{
    for(int j = 0; j < height; j++) {
        for(int i = 0; i < width; i++) {
            coordinates.at<cv::Vec<float, 4> >(j, i) = cv::Vec<float, 4>(float(i), float(j), 1.f, 0.f);
        }
    }
}

void LEDepthEstimator::initializeRandomPlane(epi_type type)
{
    for(int j = M; j < height - 2*M; j++) {
        for(int i = M; i < width - 2*M; i++) {
            this->current_plane_label[type].at<Plane>(j, i) = this->createRandomLabel(cv::Point(i, j));
        }
    }
}

void LEDepthEstimator::initializeSmoothnessCoeff()
{
    smoothness_coeff.resize(neighbours.size());
    cv::Mat central_image;
    light_field->getImage(light_field->s()/2, light_field->t()/2, central_image);

    cv::Rect inner_rect(M, M, width-2*M, height-2*M);
    cv::Mat image_p = central_image(inner_rect);

    for(int i = 0; i < neighbours.size(); i++) {
        cv::Mat image_q = central_image(inner_rect + neighbours[i]).clone();
        cv::absdiff(image_q, image_p, image_q);

        cv::exp(-cvutils::channelSum(image_q)/10.0f, smoothness_coeff[i]);
        cv::copyMakeBorder(smoothness_coeff[i], smoothness_coeff[i], M, M, M, M, cv::BORDER_CONSTANT, cv::Scalar(0));

    }
}

void LEDepthEstimator::initializeCurrentCostsFast(epi_type type)
{
    if(type == HORIZ) {
        Region& region = regions[0];
        cv::Point offset(M, M);

        cv::Mat validMask = cv::Mat::zeros(this->height, this->width, CV_8U);

        for (int j = 0; j < region.unitRegions.size(); j++){
            cv::Rect unit = region.unitRegions[j];

            int n = cv::theRNG().uniform(0, unit.height * unit.width);
            int xx = n % unit.width;
            int yy = n / unit.width;

            cv::Point pt(unit.x + xx, unit.y + yy);
            pt += offset;

            Plane label = createRandomLabel(pt);
            this->current_plane_label[HORIZ](unit + offset) = label.toScalar();

            cv::Mat validRegion = this->isValidLabel(label, unit + offset);

            validRegion.copyTo(validMask(unit + offset));
        }

        uint16_t s_hat = this->light_field->s()/2;

        computeDisparityFromPlane(HORIZ);

        #pragma omp parallel for
        for(uint16_t v = M; v < this->height - 2*M; v++) {
            for(uint16_t u = M; u < this->width - 2*M; u++) {
                /*
                 * For each pixel p = (u, v) compute the set of allowed Radiances;
                 */
                if(validMask.at<uchar>(v, u) == 0) {
                    this->current_cost[HORIZ].at<float>(v, u) = INVALID_COST;
                    continue;
                }

                float d_u = this->current_disp[HORIZ].at<float>(v, u);
                cv::Point2f centre(u, s_hat);

                double curr_cost = this->getCostFromExtendedSet(centre, HORIZ, d_u,
                                                                this->h_epi_arr[v],
                                                                this->h_epi_grad_arr[v]);

                this->current_cost[HORIZ].at<float>(v, u) = curr_cost;
            }
        }

    } else if(type == VERT) {
        Region& region = regions[0];
        cv::Point offset(M, M);

        cv::Mat validMask = cv::Mat::zeros(this->height, this->width, CV_8U);

        for (int j = 0; j < region.unitRegions.size(); j++){
            cv::Rect unit = region.unitRegions[j];

            int n = cv::theRNG().uniform(0, unit.height * unit.width);
            int xx = n % unit.width;
            int yy = n / unit.width;

            cv::Point pt(unit.x + xx, unit.y + yy);
            pt += offset;

            Plane label = createRandomLabel(pt);
            this->current_plane_label[VERT](unit + offset) = label.toScalar();

            cv::Mat validRegion = this->isValidLabel(label, unit + offset);

            validRegion.copyTo(validMask(unit + offset));
        }

        uint16_t t_hat = this->light_field->s()/2;
        computeDisparityFromPlane(VERT);

        #pragma omp parallel for
        for(uint16_t u = M; u < this->width - 2*M; u++) {
            for(uint16_t v = M; v < this->height - 2*M; v++) {
                /*
                 * For each pixel p = (u, v) compute the set of allowed Radiances;
                 */
                if(validMask.at<uchar>(v, u) == 0) {
                    this->current_cost[VERT].at<float>(v, u) = INVALID_COST;
                    continue;
                }
                float d_v = this->current_disp[VERT].at<float>(v, u);
                cv::Point2f centre(t_hat, v);

                double curr_cost = this->getCostFromExtendedSet(centre, VERT, d_v,
                                                                this->v_epi_arr[u],
                                                                this->v_epi_grad_arr[u]);

                this->current_cost[VERT].at<float>(v, u) = curr_cost;
            }
        }
    }
}

void LEDepthEstimator::initializeCurrentCosts(epi_type type)
{
    if(type == HORIZ) {
        uint16_t s_hat = light_field->s()/2;
        this->computeDisparityFromPlane(HORIZ);

//        #pragma omp parallel for
        for(uint16_t v = M; v < this->height - 2*M; v++) {
            for(uint16_t u = M; u < this->width - 2*M; u++) {
                /*
                 * For each pixel p = (u, v) compute the set of allowed Radiances;
                 */
                float d_u = this->current_disp[HORIZ].at<float>(v, u);
                cv::Point2f centre(u, s_hat);

                double curr_cost = this->getCostFromExtendedSet(centre, HORIZ, d_u,
                                                                this->h_epi_arr[v],
                                                                this->h_epi_grad_arr[v]);
                if(this->isValidLabel(this->current_plane_label[HORIZ].at<Plane>(v, u), cv::Point(u, v)))
                    this->current_cost[HORIZ].at<float>(v, u) = curr_cost;
                else
                    this->current_cost[HORIZ].at<float>(v, u) = INVALID_COST;
            }
        }
    } else if(type == VERT) {
        uint16_t t_hat = light_field->t()/2;
        this->computeDisparityFromPlane(VERT);

//        #pragma omp parallel for
        for(uint16_t u = M; u < this->width - 2*M; u++) {
            for(uint16_t v = M; v < this->height - 2*M; v++) {
                /**
                  * For each pixel p = (u, v) compute the set of allowed Radiances;
                  */
                float d_v = this->current_disp[VERT].at<float>(v, u);
                cv::Point2f centre(t_hat, v);

                double curr_cost = this->getCostFromExtendedSet(centre, VERT, d_v,
                                                                this->v_epi_arr[u],
                                                                this->v_epi_grad_arr[u]);

                if(this->isValidLabel(this->current_plane_label[VERT].at<Plane>(v, u), cv::Point(u, v)))
                    this->current_cost[VERT].at<float>(v, u) = curr_cost;
                else
                    this->current_cost[VERT].at<float>(v, u) = INVALID_COST;
            }
        }
    }
}

void LEDepthEstimator::computeDisparityFromPlane(epi_type type)
{
    this->current_disp[type] = cvutils::channelDot(this->coordinates, this->current_plane_label[type]);
}

float LEDepthEstimator::getDisparityPerturbationWidth(int iter)
{
    return (this->max_disp - this->min_disp) * pow(0.5f, iter + 1);
}

void LEDepthEstimator::tests()
{

}

void LEDepthEstimator::run()
{

    std::chrono::time_point t_0 = std::chrono::high_resolution_clock::now();
    std::cout<<"Prefetching EPI Data..."<<std::endl;
    std::chrono::time_point t_1 = std::chrono::high_resolution_clock::now();
    std::cout<<"Prefetching EPI Data took : "<<std::chrono::duration_cast<std::chrono::seconds>(t_1 - t_0).count()<<" seconds"<<std::endl;

    int pmInit = 1;

    for(int iteration = 0; iteration < pmInit; iteration++) {
        std::cout<<"Algorithm ongoing, iteration : "<<iteration<<std::endl;
        std::chrono::time_point t_i = std::chrono::high_resolution_clock::now();
        for(int grid = 0; grid < regions.size(); grid++) {
            runHorizontalRegionPropagation(iteration, grid);
            runVerticalRegionPropagation(iteration, grid);
        }
        std::chrono::time_point t_j = std::chrono::high_resolution_clock::now();
        std::cout<<"Time taken : "<<std::chrono::duration_cast<std::chrono::seconds>(t_j  - t_i).count()<<" seconds"<<std::endl;
        evaluate(iteration);
    }
    correctMarginLabels();

    int maxIter = 1;
    for(int iteration = 0; iteration < maxIter; iteration++) {
        std::cout<<"Algorithm ongoing, iteration : "<<iteration+pmInit<<std::endl;
        std::chrono::time_point t_i = std::chrono::high_resolution_clock::now();
        for(int grid = 0; grid < regions.size(); grid++) {
            runHorizontalRegionPropagation(iteration, grid);
            runVerticalRegionPropagation(iteration, grid);
        }
        std::chrono::time_point t_j = std::chrono::high_resolution_clock::now();
        std::cout<<"Time taken : "<<std::chrono::duration_cast<std::chrono::seconds>(t_j  - t_i).count()<<" seconds"<<std::endl;
        evaluate(iteration + pmInit);
    }
    std::chrono::time_point t_n = std::chrono::high_resolution_clock::now();
    std::cout<<"Total running time : "<<std::chrono::duration_cast<std::chrono::seconds>(t_n - t_0).count()<<" seconds"<<std::endl;

    std::cout<<"Post processing :"<<std::endl;
    postProcess();

}

bool LEDepthEstimator::isValidLabel(Plane label, cv::Point pos)
{
    float ds = label.GetZ(pos);
    float a5 = label.a * 5;
    float b5 = label.b * 5;
    float d;

    return (
        ds >= this->min_disp && ds <= this->max_disp
        && ((d = ds + a5 + b5) >= this->min_disp) && d <= this->max_disp
        && ((d = ds + a5 - b5) >= this->min_disp) && d <= this->max_disp
        && ((d = ds - a5 + b5) >= this->min_disp) && d <= this->max_disp
        && ((d = ds - a5 - b5) >= this->min_disp) && d <= this->max_disp
    );
}

cv::Mat LEDepthEstimator::isValidLabel(Plane label, cv::Rect rect)
{
    float a5 = label.a * 5;
    float b5 = label.b * 5;
    const cv::Scalar lower(this->min_disp);
    const cv::Scalar upper(this->max_disp);
    cv::Mat mask = cv::Mat::zeros(rect.size(), CV_8U);
    cv::Mat disp = cvutils::channelSum(this->coordinates(rect).mul(label.toScalar()));
    for (int y = 0; y < mask.rows; y++)
    {
        for (int x = 0; x < mask.cols; x++)
        {
            float ds = disp.at<float>(y, x);
            float d;

            mask.at<uchar>(y, x) = (ds >= this->min_disp && ds <= this->max_disp
                && ((d = ds + a5 + b5) >= this->min_disp) && d <= this->max_disp
                && ((d = ds + a5 - b5) >= this->min_disp) && d <= this->max_disp
                && ((d = ds - a5 + b5) >= this->min_disp) && d <= this->max_disp
                && ((d = ds - a5 - b5) >= this->min_disp) && d <= this->max_disp
                ) ? 255 : 0;
        }
    }
    return mask;
}

void LEDepthEstimator::prefetchEPIData(bool archive)
{
    if(archive) {

        uint16_t s_hat = light_field->s()/2;
        uint16_t t_hat = light_field->t()/2;
        this->h_epi_arr.resize(this->height);
        this->h_epi_grad_arr.resize(this->height);

        omp_lock_t lck;
        omp_init_lock(&lck);

        const int ksize = 1;
        const double scale = 0.5;

        #pragma omp parallel for
        for(uint16_t v = 0; v < this->height; v++) {
            cv::Mat h_epi;
            omp_set_lock(&lck);
            light_field->getEPIVT(v, t_hat, h_epi);
            omp_unset_lock(&lck);

            CV_Assert(h_epi.cols == this->width);

            cv::Mat h_epi_eq;
            cv::Mat channels[3];
            cv::split(h_epi, channels);
            for(int i = 0; i < 3; i++)
                cv::equalizeHist(channels[i], channels[i]);

            cv::merge(channels, 3, h_epi_eq);

            h_epi_eq.convertTo(h_epi_eq, CV_32F);

            cv::Mat h_epi_grad;
            cv::Sobel(h_epi_eq, h_epi_grad, CV_32F, 1, 0, ksize, scale, 0, cv::BORDER_REPLICATE);

            this->h_epi_arr[v] = h_epi_eq * (1.f - alpha);
            this->h_epi_grad_arr[v] = h_epi_grad * alpha;

        }


        this->v_epi_arr.resize(this->width);
        this->v_epi_grad_arr.resize(this->width);

        #pragma omp parallel for
        for(uint16_t u = 0; u < this->width; u++) {
            cv::Mat v_epi;
            omp_set_lock(&lck);
            light_field->getEPIUS(u, s_hat, v_epi);
            omp_unset_lock(&lck);

            CV_Assert(v_epi.rows == this->height);

            cv::Mat v_epi_eq;
            cv::Mat channels[3];
            cv::split(v_epi, channels);
            for(int i = 0; i < 3; i++)
                cv::equalizeHist(channels[i], channels[i]);

            cv::merge(channels, 3, v_epi_eq);

            v_epi_eq.convertTo(v_epi_eq, CV_32F);

            cv::Mat v_epi_grad;
            cv::Sobel(v_epi_eq, v_epi_grad, CV_32F, 0, 1, ksize, scale, 0, cv::BORDER_REPLICATE);

            this->v_epi_arr[u] = v_epi_eq * (1.f - alpha);
            this->v_epi_grad_arr[u] = v_epi_grad * alpha;

        }

        omp_destroy_lock(&lck);

        this->serializeEPIData();

    } else {
        this->deserializeEPIData();
    }
}

void LEDepthEstimator::runHorizontalRegionPropagation(int iter, int grid, bool do_gc)
{

    Region& region = regions[grid];
    // 16-time loop
    for(int j = 0; j < region.disjointRegionSets.size(); j++) {

        runHorizontalIterativeExpansion(iter, grid, j, do_gc);

    }

}

void LEDepthEstimator::runVerticalRegionPropagation(int iter, int grid, bool do_gc)
{
    Region& region = regions[grid];
    // 16-time loop
    for(int j = 0; j < region.disjointRegionSets.size(); j++) {

        runVerticalIterativeExpansion(iter, grid, j, do_gc);

    }
}

void LEDepthEstimator::runHorizontalIterativeExpansion(int iter, int grid, int set, bool do_gc)
{
    if(grid == 0) {

        Region& region = regions[grid];

        int K_exp = 1;
        int K_ransac = 1;
        int K_random = 7;

        int innerIter = 0;
        while(innerIter++ < K_exp) {
            runHorizontalExpansionProposer(region, set, do_gc);
        }

        innerIter = 0;
        while(innerIter++ < K_ransac) {
            runHorizontalRansacProposer(region, set, do_gc);
        }

        innerIter = iter;
        while(innerIter < K_random && getDisparityPerturbationWidth(innerIter) < 0.1) {
            runHorizontalRandomProposer(region, set, innerIter++, do_gc);
        }


    } else if (grid == 1) {

        Region& region = regions[grid];

        int K_exp = 2;
        int K_ransac = 1;

        int innerIter = 0;
        while(innerIter++ < K_exp) {
            runHorizontalExpansionProposer(region, set, do_gc);
        }

        innerIter = 0;
        while(innerIter++ < K_ransac) {
            runHorizontalRansacProposer(region, set, do_gc);
        }


    } else if (grid == 2) {

        Region& region = regions[grid];

        int K_exp = 2;
        int K_ransac = 1;

        int innerIter = 0;
        while(innerIter++ < K_exp) {
            runHorizontalExpansionProposer(region, set, do_gc);
        }

        innerIter = 0;
        while(innerIter++ < K_ransac) {
            runHorizontalRansacProposer(region, set, do_gc);
        }

    }


}

void LEDepthEstimator::runVerticalIterativeExpansion(int iter, int grid, int set, bool do_gc)
{
    if(grid == 0) {

        Region& region = regions[grid];

        int K_exp = 1;
        int K_ransac = 1;
        int K_random = 7;

        int innerIter = 0;
        while(innerIter++ < K_exp) {
            runVerticalExpansionProposer(region, set, do_gc);
        }

        innerIter = 0;
        while(innerIter++ < K_ransac) {
            runVerticalRansacProposer(region, set, do_gc);
        }

        innerIter = iter;
        while(innerIter < K_random && getDisparityPerturbationWidth(innerIter) < 0.1) {
            runVerticalRandomProposer(region, set, innerIter++, do_gc);
        }


    } else if (grid == 1) {

        Region& region = regions[grid];

        int K_exp = 2;
        int K_ransac = 1;

        int innerIter = 0;
        while(innerIter++ < K_exp) {
            runVerticalExpansionProposer(region, set, do_gc);
        }

        innerIter = 0;
        while(innerIter++ < K_ransac) {
            runVerticalRansacProposer(region, set, do_gc);
        }

    } else if (grid == 2) {

        Region& region = regions[grid];

        int K_exp = 2;
        int K_ransac = 1;

        int innerIter = 0;
        while(innerIter++ < K_exp) {
            runVerticalExpansionProposer(region, set, do_gc);
        }

        innerIter = 0;
        while(innerIter++ < K_ransac) {
            runVerticalRansacProposer(region, set, do_gc);
        }

    }
}

void LEDepthEstimator::runHorizontalExpansionProposer(const Region& region, int set, bool do_gc)
{
    cv::Point offset(M, M);
    cv::Mat validMask = cv::Mat::zeros(this->height, this->width, CV_8U);
    cv::Mat pixelMask = cv::Mat::zeros(this->height, this->width, CV_8U);
    cv::Mat proposedLabels = cv::Mat::zeros(this->height, this->width, cv::DataType<Plane>::type);

    // Can be parallized here
    for(int n = 0; n <region.disjointRegionSets[set].size(); n++) {

        int r = region.disjointRegionSets[set][n];

        const cv::Rect& unitRegion = region.unitRegions[r];
        const cv::Rect& sharedRegion = region.sharedRegions[r];

        Plane alpha = getExpansionPlane(unitRegion + offset, HORIZ);

        proposedLabels(sharedRegion + offset).setTo(alpha.toScalar());
        cv::Mat validRegion = this->isValidLabel(alpha, sharedRegion + offset);

        validRegion.copyTo(validMask(sharedRegion + offset));
        pixelMask(sharedRegion + offset).setTo(255);
    }

    cv::Mat proposalCosts = cv::Mat::zeros(this->height, this->width, CV_32F);

    runHorizontalCostComputation(proposalCosts, proposedLabels, pixelMask, validMask);

    #pragma omp parallel for
    for(int n = 0; n < region.disjointRegionSets[set].size(); n++) {
        int r = region.disjointRegionSets[set][n];

        const cv::Rect& sharedRegion = region.sharedRegions[r];
        Plane alpha = proposedLabels(sharedRegion + offset).at<Plane>(0, 0);
        const cv::Mat& localProposals = proposalCosts(sharedRegion + offset);
        cv::Mat updateMask = cv::Mat::zeros(sharedRegion.size(), CV_8U);

        if(do_gc)
            runGCExpansion(sharedRegion + offset, localProposals, alpha, updateMask);
        else
            updateMask = this->current_cost[HORIZ](sharedRegion+offset) > proposalCosts(sharedRegion + offset);

        proposalCosts(sharedRegion + offset).copyTo(this->current_cost[HORIZ](sharedRegion + offset), updateMask);
        this->current_plane_label[HORIZ](sharedRegion + offset).setTo(alpha.toScalar(), updateMask);
    }
}

void LEDepthEstimator::runHorizontalRansacProposer(const Region &region, int set, bool do_gc)
{
    cv::Point offset(M, M);

    cv::Mat validMask = cv::Mat::zeros(this->height, this->width, CV_8U);
    cv::Mat pixelMask = cv::Mat::zeros(this->height, this->width, CV_8U);
    cv::Mat proposedLabels = cv::Mat::zeros(this->height, this->width, cv::DataType<Plane>::type);

    this->computeDisparityFromPlane(HORIZ);

    for(int n = 0; n <region.disjointRegionSets[set].size(); n++) {

        int r = region.disjointRegionSets[set][n];

        const cv::Rect& unitRegion = region.unitRegions[r];
        const cv::Rect& sharedRegion = region.sharedRegions[r];

        Plane alpha = getRANSACPlane(unitRegion + offset, HORIZ);

        proposedLabels(sharedRegion + offset).setTo(alpha.toScalar());
        cv::Mat validRegion = this->isValidLabel(alpha, sharedRegion + offset);

        validRegion.copyTo(validMask(sharedRegion + offset));
        pixelMask(sharedRegion + offset).setTo(255);
    }

    cv::Mat proposalCosts = cv::Mat::zeros(this->height, this->width, CV_32F);

    runHorizontalCostComputation(proposalCosts, proposedLabels, pixelMask, validMask);

    #pragma omp parallel for
    for(int n = 0; n < region.disjointRegionSets[set].size(); n++) {
        int r = region.disjointRegionSets[set][n];

        const cv::Rect& sharedRegion = region.sharedRegions[r];
        Plane alpha = proposedLabels(sharedRegion + offset).at<Plane>(0, 0);
        const cv::Mat& localProposals = proposalCosts(sharedRegion + offset);
        cv::Mat updateMask = cv::Mat::zeros(sharedRegion.size(), CV_8U);

        if(do_gc)
            runGCExpansion(sharedRegion + offset, localProposals, alpha, updateMask);
        else
            updateMask = this->current_cost[HORIZ](sharedRegion+offset) > proposalCosts(sharedRegion + offset);

        proposalCosts(sharedRegion + offset).copyTo(this->current_cost[HORIZ](sharedRegion + offset), updateMask);
        this->current_plane_label[HORIZ](sharedRegion + offset).setTo(alpha.toScalar(), updateMask);
    }
}

void LEDepthEstimator::runHorizontalRandomProposer(const Region &region, int iter, int set, bool do_gc)
{
    cv::Point offset(M, M);

    cv::Mat validMask = cv::Mat::zeros(this->height, this->width, CV_8U);
    cv::Mat pixelMask = cv::Mat::zeros(this->height, this->width, CV_8U);
    cv::Mat proposedLabels = cv::Mat::zeros(this->height, this->width, cv::DataType<Plane>::type);

    //Parallelize this
    for(int n = 0; n <region.disjointRegionSets[set].size(); n++) {
        int r = region.disjointRegionSets[set][n];

        const cv::Rect& unitRegion = region.unitRegions[r];
        const cv::Rect& sharedRegion = region.sharedRegions[r];

        Plane alpha = getRandomPlane(unitRegion + offset, HORIZ, iter);

        proposedLabels(sharedRegion + offset).setTo(alpha.toScalar());
        cv::Mat validRegion = this->isValidLabel(alpha, sharedRegion + offset);

        validRegion.copyTo(validMask(sharedRegion + offset));
        pixelMask(sharedRegion + offset).setTo(255);
    }

    cv::Mat proposalCosts = cv::Mat::zeros(this->height, this->width, CV_32F);

    runHorizontalCostComputation(proposalCosts, proposedLabels, pixelMask, validMask);

    //Parallelize this
    #pragma omp parallel for
    for(int n = 0; n < region.disjointRegionSets[set].size(); n++) {
        int r = region.disjointRegionSets[set][n];

        const cv::Rect& sharedRegion = region.sharedRegions[r];
        Plane alpha = proposedLabels(sharedRegion + offset).at<Plane>(0, 0);
        const cv::Mat& localProposals = proposalCosts(sharedRegion + offset);
        cv::Mat updateMask = cv::Mat::zeros(sharedRegion.size(), CV_8U);

        if(do_gc)
            runGCExpansion(sharedRegion + offset, localProposals, alpha, updateMask);
        else
            updateMask = this->current_cost[HORIZ](sharedRegion+offset) > proposalCosts(sharedRegion + offset);

        proposalCosts(sharedRegion + offset).copyTo(this->current_cost[HORIZ](sharedRegion + offset), updateMask);
        this->current_plane_label[HORIZ](sharedRegion + offset).setTo(alpha.toScalar(), updateMask);
    }
}

void LEDepthEstimator::runVerticalExpansionProposer(const Region &region, int set, bool do_gc)
{
    cv::Point offset(M, M);
    cv::Mat validMask = cv::Mat::zeros(this->height, this->width, CV_8U);
    cv::Mat pixelMask = cv::Mat::zeros(this->height, this->width, CV_8U);
    cv::Mat proposedLabels = cv::Mat::zeros(this->height, this->width, cv::DataType<Plane>::type);

    // Can be parallized here
    for(int n = 0; n <region.disjointRegionSets[set].size(); n++) {

        int r = region.disjointRegionSets[set][n];

        const cv::Rect& unitRegion = region.unitRegions[r];
        const cv::Rect& sharedRegion = region.sharedRegions[r];

        Plane alpha = getExpansionPlane(unitRegion + offset, VERT);

        proposedLabels(sharedRegion + offset).setTo(alpha.toScalar());
        cv::Mat validRegion = this->isValidLabel(alpha, sharedRegion + offset);

        validRegion.copyTo(validMask(sharedRegion + offset));
        pixelMask(sharedRegion + offset).setTo(255);
    }

    cv::Mat proposalCosts = cv::Mat::zeros(this->height, this->width, CV_32F);

    runVerticalCostComputation(proposalCosts, proposedLabels, pixelMask, validMask);

    //Parallelize this
    #pragma omp parallel for
    for(int n = 0; n < region.disjointRegionSets[set].size(); n++) {
        int r = region.disjointRegionSets[set][n];

        const cv::Rect& sharedRegion = region.sharedRegions[r];
        Plane alpha = proposedLabels(sharedRegion + offset).at<Plane>(0, 0);
        const cv::Mat& localProposals = proposalCosts(sharedRegion + offset);
        cv::Mat updateMask = cv::Mat::zeros(sharedRegion.size(), CV_8U);

        if(do_gc)
            runGCExpansion(sharedRegion + offset, localProposals, alpha, updateMask);
        else
            updateMask = this->current_cost[VERT](sharedRegion+offset) > proposalCosts(sharedRegion + offset);

        proposalCosts(sharedRegion + offset).copyTo(this->current_cost[VERT](sharedRegion + offset), updateMask);
        this->current_plane_label[VERT](sharedRegion + offset).setTo(alpha.toScalar(), updateMask);
    }
}

void LEDepthEstimator::runVerticalRansacProposer(const Region &region, int set, bool do_gc)
{
    cv::Point offset(M, M);
    cv::Mat validMask = cv::Mat::zeros(this->height, this->width, CV_8U);
    cv::Mat pixelMask = cv::Mat::zeros(this->height, this->width, CV_8U);
    cv::Mat proposedLabels = cv::Mat::zeros(this->height, this->width, cv::DataType<Plane>::type);

    this->computeDisparityFromPlane(VERT);

    // Can be parallized here
    for(int n = 0; n <region.disjointRegionSets[set].size(); n++) {

        int r = region.disjointRegionSets[set][n];

        const cv::Rect& unitRegion = region.unitRegions[r];
        const cv::Rect& sharedRegion = region.sharedRegions[r];

        Plane alpha = getRANSACPlane(unitRegion + offset, VERT);

        proposedLabels(sharedRegion + offset).setTo(alpha.toScalar());
        cv::Mat validRegion = this->isValidLabel(alpha, sharedRegion + offset);

        validRegion.copyTo(validMask(sharedRegion + offset));
        pixelMask(sharedRegion + offset).setTo(255);
    }

    cv::Mat proposalCosts = cv::Mat::zeros(this->height, this->width, CV_32F);

    runVerticalCostComputation(proposalCosts, proposedLabels, pixelMask, validMask);

    //Parallelize this
    #pragma omp parallel for
    for(int n = 0; n < region.disjointRegionSets[set].size(); n++) {
        int r = region.disjointRegionSets[set][n];

        const cv::Rect& sharedRegion = region.sharedRegions[r];
        Plane alpha = proposedLabels(sharedRegion + offset).at<Plane>(0, 0);
        const cv::Mat& localProposals = proposalCosts(sharedRegion + offset);
        cv::Mat updateMask = cv::Mat::zeros(sharedRegion.size(), CV_8U);

        if(do_gc)
            runGCExpansion(sharedRegion + offset, localProposals, alpha, updateMask);
        else
            updateMask = this->current_cost[VERT](sharedRegion+offset) > proposalCosts(sharedRegion + offset);

        proposalCosts(sharedRegion + offset).copyTo(this->current_cost[VERT](sharedRegion + offset), updateMask);
        this->current_plane_label[VERT](sharedRegion + offset).setTo(alpha.toScalar(), updateMask);
    }
}

void LEDepthEstimator::runVerticalRandomProposer(const Region &region, int iter, int set, bool do_gc)
{
    cv::Point offset(M, M);
    cv::Mat validMask = cv::Mat::zeros(this->height, this->width, CV_8U);
    cv::Mat pixelMask = cv::Mat::zeros(this->height, this->width, CV_8U);
    cv::Mat proposedLabels = cv::Mat::zeros(this->height, this->width, cv::DataType<Plane>::type);

    // Can be parallized here
    for(int n = 0; n <region.disjointRegionSets[set].size(); n++) {

        int r = region.disjointRegionSets[set][n];

        const cv::Rect& unitRegion = region.unitRegions[r];
        const cv::Rect& sharedRegion = region.sharedRegions[r];

        Plane alpha = getRandomPlane(unitRegion + offset, VERT, iter);

        proposedLabels(sharedRegion + offset).setTo(alpha.toScalar());
        cv::Mat validRegion = this->isValidLabel(alpha, sharedRegion + offset);

        validRegion.copyTo(validMask(sharedRegion + offset));
        pixelMask(sharedRegion + offset).setTo(255);
    }

    cv::Mat proposalCosts = cv::Mat::zeros(this->height, this->width, CV_32F);

    runVerticalCostComputation(proposalCosts, proposedLabels, pixelMask, validMask);

    //Parallelize this
    #pragma omp parallel for
    for(int n = 0; n < region.disjointRegionSets[set].size(); n++) {
        int r = region.disjointRegionSets[set][n];

        const cv::Rect& sharedRegion = region.sharedRegions[r];
        Plane alpha = proposedLabels(sharedRegion + offset).at<Plane>(0, 0);
        const cv::Mat& localProposals = proposalCosts(sharedRegion + offset);
        cv::Mat updateMask = cv::Mat::zeros(sharedRegion.size(), CV_8U);

        if(do_gc)
            runGCExpansion(sharedRegion + offset, localProposals, alpha, updateMask);
        else
            updateMask = this->current_cost[VERT](sharedRegion+offset) > proposalCosts(sharedRegion + offset);

        proposalCosts(sharedRegion + offset).copyTo(this->current_cost[VERT](sharedRegion + offset), updateMask);
        this->current_plane_label[VERT](sharedRegion + offset).setTo(alpha.toScalar(), updateMask);
    }
}

void LEDepthEstimator::runHorizontalCostComputation(cv::Mat &proposalCosts, cv::Mat& proposedLabels, cv::Mat& pixelMask, cv::Mat& validMask)
{
    int s_hat = light_field->s()/2;

    #pragma omp parallel for
    for(uint16_t v = M; v < this->height - 2*M; v++) {
        for(uint16_t u = M; u < this->width - 2*M; u++) {
            cv::Point img_point(u, v);

            if(pixelMask.at<uchar>(img_point) == 0) continue;
            if(validMask.at<uchar>(img_point) == 0) {
                proposalCosts.at<float>(img_point) = INVALID_COST;
                continue;
            }

            /*
             * For each pixel p = (u, v) compute the set of allowed Radiances;
             */
            float d_u = proposedLabels.at<Plane>(v, u).GetZ(img_point);
            cv::Point2f centre(u, s_hat);

            double curr_cost = this->getCostFromExtendedSet(centre, HORIZ, d_u,
                                                            this->h_epi_arr[v],
                                                            this->h_epi_grad_arr[v]);

            proposalCosts.at<float>(img_point) = curr_cost;
        }
    }

}

void LEDepthEstimator::runVerticalCostComputation(cv::Mat &proposalCosts, cv::Mat &proposedLabels, cv::Mat &pixelMask, cv::Mat &validMask)
{
    int t_hat = light_field->t()/2;

    #pragma omp parallel for
    for(uint16_t u = M; u < this->width - 2*M; u++) {
        for(uint16_t v = M; v < this->height - 2*M; v++) {
            cv::Point img_point(u, v);

            if(pixelMask.at<uchar>(img_point) == 0) continue;
            if(validMask.at<uchar>(img_point) == 0) {
                proposalCosts.at<float>(img_point) = INVALID_COST;
                continue;
            }
            /*
             * For each pixel p = (u, v) compute the set of allowed Radiances;
             */
            float d_v = proposedLabels.at<Plane>(v, u).GetZ(img_point);
            cv::Point2f centre(t_hat, v);

            double curr_cost = this->getCostFromExtendedSet(centre, VERT, d_v,
                                                            this->v_epi_arr[u],
                                                            this->v_epi_grad_arr[u]);

            proposalCosts.at<float>(img_point) = curr_cost;
        }
    }
}

void LEDepthEstimator::runGCExpansion(const cv::Rect &sharedRegion, const cv::Mat &localProposals, Plane alpha, cv::Mat &updateMask)
{

    std::vector<cv::Mat> cost_fp_fq(neighbours.size());
    std::vector<cv::Mat> cost_fp_alpha(neighbours.size());
    std::vector<cv::Mat> cost_alpha_fq(neighbours.size());

    cv::Mat fp = this->current_plane_label[HORIZ](sharedRegion);
    cv::Mat coord_p = this->coordinates(sharedRegion);

    cv::Mat dp_fp = cvutils::channelDot(fp, coord_p);
    cv::Mat dp_alpha = cvutils::channelSum(coord_p.mul(alpha.toScalar()));
    for(int i = 0; i < neighbours.size(); i++) {

        cv::Rect neighbourUnitRegion = sharedRegion + neighbours[i];

        cv::Mat fq = this->current_plane_label[HORIZ](neighbourUnitRegion);
        cv::Mat coord_q = this->coordinates(neighbourUnitRegion);

        cv::Mat dq_fq = cvutils::channelDot(fq, coord_q);
        cv::Mat dq_alpha = cvutils::channelSum(coord_q.mul(alpha.toScalar()));

        cv::Mat dp_fq = cvutils::channelDot(fq, coord_p);
        cv::Mat dq_fp = cvutils::channelDot(fp, coord_q);

        cv::Mat smoothness_term = this->smoothness_coeff[i](sharedRegion);

        float param_lambda = 1.0f;

        cost_fp_fq[i] = cv::abs(dp_fp - dp_fq) + cv::abs(dq_fq - dq_fp);
        cv::threshold(cost_fp_fq[i], cost_fp_fq[i], thresh_smooth, 0, cv::THRESH_TRUNC);
        cost_fp_fq[i] = cost_fp_fq[i].mul(smoothness_term, param_lambda);

        cost_fp_alpha[i] = cv::abs(dp_fp - dp_alpha) + cv::abs(dq_alpha - dq_fp);
        cv::threshold(cost_fp_alpha[i], cost_fp_alpha[i], thresh_smooth, 0, cv::THRESH_TRUNC);
        cost_fp_alpha[i] = cost_fp_alpha[i].mul(smoothness_term, param_lambda);

        cost_alpha_fq[i] = cv::abs(dp_alpha - dp_fq) + cv::abs(dq_fq - dq_alpha);
        cv::threshold(cost_alpha_fq[i], cost_alpha_fq[i], thresh_smooth, 0 , cv::THRESH_TRUNC);
        cost_alpha_fq[i] = cost_alpha_fq[i].mul(smoothness_term, param_lambda);

    }

    int n = sharedRegion.width * sharedRegion.height;
    typedef Graph<float, float, double> G;
    G graph(n, 4 * n);

    graph.add_node(n);
    cv::Point sharedRegion_origin(sharedRegion.x, sharedRegion.y);

//    cv::Point offset(M, M);

    for(int y = 0; y < sharedRegion.height; y++) {
        for(int x = 0; x < sharedRegion.width; x++) {
            int i = y*sharedRegion.width + x;
            cv::Point sharedRegion_coord(x, y);
            cv::Point imageRegion_coord = sharedRegion_coord + sharedRegion_origin;
            graph.add_tweights(i, this->current_cost[HORIZ].at<float>(imageRegion_coord), localProposals.at<float>(sharedRegion_coord));

            bool x0 = x == 0;
            bool x1 = x == sharedRegion.width - 1;
            bool y0 = y == 0;
            bool y1 = y == sharedRegion.height - 1;

            if(x0)
                graph.add_tweights(i, cost_fp_fq[W].at<float>(y, x), cost_alpha_fq[W].at<float>(y, x));
            if(x1)
                graph.add_tweights(i, cost_fp_fq[E].at<float>(y, x), cost_alpha_fq[E].at<float>(y, x));
            if(y0)
                graph.add_tweights(i, cost_fp_fq[N].at<float>(y, x), cost_alpha_fq[N].at<float>(y, x));
            if(y1)
                graph.add_tweights(i, cost_fp_fq[S].at<float>(y, x), cost_alpha_fq[S].at<float>(y, x));

            if(x0 && y0)
                graph.add_tweights(i, cost_fp_fq[NW].at<float>(y, x), cost_alpha_fq[NW].at<float>(y, x));
            if(x1 && y0)
                graph.add_tweights(i, cost_fp_fq[NE].at<float>(y, x), cost_alpha_fq[NE].at<float>(y, x));
            if(x0 && y1)
                graph.add_tweights(i, cost_fp_fq[SW].at<float>(y, x), cost_alpha_fq[SW].at<float>(y, x));
            if(x1 && y1)
                graph.add_tweights(i, cost_fp_fq[SE].at<float>(y, x), cost_alpha_fq[SE].at<float>(y, x));

        }
    }

    /**
  *			NW	N	NE
  * 		W	C	E
  * 		SW	S	SE
  *
  * 	C <=> E
  */
    for(int y = 0; y < sharedRegion.height; y++) {
        for(int x = 0; x < sharedRegion.width - 1; x++) {
            int i = y*sharedRegion.width + x;
            int j = y*sharedRegion.width + x + 1;

            float B = cost_alpha_fq[E].at<float>(y, x);
            float C = cost_fp_alpha[E].at<float>(y, x);
            float D = cost_fp_fq[E].at<float>(y, x);

            graph.add_edge(i, j, std::max(0.f, B + C - D), 0);
            graph.add_tweights(i, C, 0);
            graph.add_tweights(j, D - C, 0);

        }
    }

    /**
  *		C <=> S
  */
    for(int y = 0; y < sharedRegion.height - 1; y++) {
        for(int x = 0; x < sharedRegion.width; x++) {
            int i = y*sharedRegion.width + x;
            int j = (y + 1)*sharedRegion.width + x;

            float B = cost_alpha_fq[S].at<float>(y, x);
            float C = cost_fp_alpha[S].at<float>(y, x);
            float D = cost_fp_fq[S].at<float>(y, x);

            graph.add_edge(i, j, std::max(0.f, B + C - D), 0);
            graph.add_tweights(i, C, 0);
            graph.add_tweights(j, D - C, 0);

        }
    }

    /**
  *		C <=> SE
  */
    for(int y = 0; y < sharedRegion.height - 1; y++) {
        for(int x = 0; x < sharedRegion.width; x++) {
            int i = y*sharedRegion.width + x;
            int j = (y + 1)*sharedRegion.width + x - 1;

            float B = cost_alpha_fq[SE].at<float>(y, x);
            float C = cost_fp_alpha[SE].at<float>(y, x);
            float D = cost_fp_fq[SE].at<float>(y, x);

            graph.add_edge(i, j, std::max(0.f, B + C - D), 0);
            graph.add_tweights(i, C, 0);
            graph.add_tweights(j, D - C, 0);

        }
    }

    /**
  *		C <=> SW
  */
    for(int y = 0; y < sharedRegion.height - 1; y++) {
        for(int x = 0; x < sharedRegion.width - 1; x++) {
            int i = y*sharedRegion.width + x;
            int j = (y + 1)*sharedRegion.width + x + 1;

            float B = cost_alpha_fq[SW].at<float>(y, x);
            float C = cost_fp_alpha[SW].at<float>(y, x);
            float D = cost_fp_fq[SW].at<float>(y, x);

            graph.add_edge(i, j, std::max(0.f, B + C - D), 0);
            graph.add_tweights(i, C, 0);
            graph.add_tweights(j, D - C, 0);

        }
    }

    double flow = graph.maxflow();

    for (int y = 0; y < sharedRegion.height; y++){
        for (int x = 0; x < sharedRegion.width; x++){
            if(graph.what_segment(y*sharedRegion.width + x) == G::SOURCE) {
                updateMask.at<uchar>(y, x) = 255;
            }
        }
    }

}

Plane LEDepthEstimator::getExpansionPlane(const cv::Rect& unitRegion, epi_type type)
{
    int n = cv::theRNG().uniform(0, unitRegion.height * unitRegion.width);

    int xx = n % unitRegion.width;
    int yy = n / unitRegion.width;

    cv::Point pt(unitRegion.x + xx, unitRegion.y + yy);

    return this->current_plane_label[type].at<Plane>(pt);
}

Plane LEDepthEstimator::getRANSACPlane(const cv::Rect& unitRegion, epi_type type)
{
    cv::Mat unitCoord = cv::Mat(unitRegion.size(), CV_32FC3);
    cv::Mat unitDisp = cv::Mat(unitRegion.size(), CV_32FC1);

    for(int y = 0; y < unitRegion.height; y++) {
        for(int x = 0; x < unitRegion.width; x++) {
            cv::Vec3f& coord = unitCoord.at<cv::Vec3f>(y, x);
            cv::Vec4f img_coord = this->coordinates.at<cv::Vec4f>(y + unitRegion.y, x + unitRegion.x);
            coord[0] = img_coord[0];
            coord[1] = img_coord[1];
            coord[2] = img_coord[2];

            unitDisp.at<float>(y, x) = this->current_disp[type].at<float>(y + unitRegion.y, x + unitRegion.x);
        }
    }

    unitCoord = unitCoord.reshape(1, unitCoord.rows * unitCoord.cols);
    unitDisp = unitDisp.reshape(1, unitDisp.rows * unitDisp.cols);

    int len = unitCoord.rows;
    int max_i = 3;
    int max_sam = 500;
    int no_sam = 0;
    cv::Mat div = cv::Mat_<float>::zeros(3, 1);
    cv::Mat inls = cv::Mat_<uchar>(len, 1, (uchar)0);
    int no_i_c = 0;
    cv::Mat N = cv::Mat_<float>(3, 1);
    cv::Mat result;
    float conf = 0.95f;
    float threshold = 1.0f;

    cv::Mat ranpts = cv::Mat_<float>::zeros(3, 3);

    while (no_sam < max_sam)
    {
        no_sam = no_sam + 1;
        std::vector<int> ransam;
        ransam.reserve(len);
        for(int i = 0; i < len; ++i)
            ransam.push_back(i);

        std::random_shuffle(ransam.begin(), ransam.end());

        for (int i = 0; i < 3; i++)
        {
            ranpts.at<cv::Vec3f>(i) = unitCoord.at<cv::Vec3f>(ransam[i]);
            div.at<float>(i) = unitDisp.at<float>(ransam[i]);
        }
        /// compute a distance of all points to a plane given by pts(:, sam) to dist
        cv::solve(ranpts, div, N, cv::DECOMP_SVD);
        cv::Mat dist = cv::abs(unitCoord * N - unitDisp);
        cv::Mat v = dist < threshold;
        int no_i = cv::countNonZero(v);

        if (max_i < no_i)
        {
            // Re - estimate plane and inliers
            cv::Mat b = cv::Mat_<float>::zeros(no_i, 1);
            cv::Mat A = cv::Mat_<float>::zeros(no_i, 3);

            // MATLAB: A = pts(v, :);
            for (int i = 0, j = 0; i < no_i; i++)
            if (v.at<uchar>(i))
            {
                A.at<cv::Vec3f>(j) = unitCoord.at<cv::Vec3f>(i);
                b.at<float>(j) = unitDisp.at<float>(i);
                j++;
            }

            cv::solve(A, b, N, cv::DECOMP_SVD);
            dist = cv::abs(unitCoord * N - unitDisp);
            v = dist < threshold;
            int no = cv::countNonZero(v);

            if (no > no_i_c)
            {
                result = N.clone();
                no_i_c = no;
                inls = v;
                max_i = no_i;
                max_sam = std::min(max_sam, this->computeSampleCount(no, len, 3, conf));
            }
        }
    }
    return Plane(result.at<float>(0), result.at<float>(1), result.at<float>(2));
}

Plane LEDepthEstimator::getRandomPlane(const cv::Rect& unitRegion, epi_type type, int iter)
{
    int n = cv::theRNG().uniform(0, unitRegion.height * unitRegion.width);

    int xx = n % unitRegion.width;
    int yy = n / unitRegion.width;

    cv::Point pt(unitRegion.x + xx, unitRegion.y + yy);

    Plane pl = this->current_plane_label[type].at<Plane>(pt);

    float zs = pl.GetZ(float(pt.x), float(pt.y));
    float dz = getDisparityPerturbationWidth(iter);
    float minz = std::max(this->min_disp, zs - dz);
    float maxz = std::min(this->max_disp, zs + dz);
    zs = cv::theRNG().uniform(minz, maxz);

    float nr = getDisparityPerturbationWidth(iter - 1);

    cv::Vec<float, 3> nv = pl.GetNormal() + (cv::Vec<float, 3>) cvutils::getRandomUnitVector() * nr;
    nv = nv / sqrt(nv.ddot(nv));

    return Plane::CreatePlane(nv, zs, float(pt.x), float(pt.y));
}

int LEDepthEstimator::computeSampleCount(int ni, int ptNum, int pf, double conf)
{
    int SampleCnt;

    double q = 1.0;
    for (double a = (ni - pf + 1), b = (ptNum - pf + 1); a <= ni; a += 1.0, b += 1.0)
        q *= (a / b);

    const double eps = 1e-4;

    if ((1.0 - q) < eps)
        SampleCnt = 1;
    else
        SampleCnt = int(log(1.0 - conf) / log(1.0 - q));

    if (SampleCnt < 1)
        SampleCnt = 1;
    return SampleCnt;
}

bool LEDepthEstimator::isPixelInSet(cv::Point pt, const Region &region, int set)
{

}

int LEDepthEstimator::getRegionIDfromPixel(cv::Point pt, const Region &region, int set)
{
    cv::Point offset(M, M);
    int regionUnitSize = region.unitRegionSize;

    int x = (pt.x - offset.x) / regionUnitSize;
    int y = (pt.y - offset.y) / regionUnitSize;

    x = x % 4;
    y = y % 4;

}

double LEDepthEstimator::getCostFromExtendedSet(const cv::Point2f &centre, epi_type type, const float disparity, const cv::Mat &epi, const cv::Mat &epi_grad)
{
    if(std::isnan(disparity)) {
        return INVALID_COST;
    }
    std::vector<cv::Point2f> pixel_set;
    this->getPixelSet(centre, pixel_set, type, disparity);
    double C_s = 0.;
    double sigma_s = 1, sigma_c = 5;

    for(int shift = -W; shift <= W; shift++) {

        cv::Point2f s_center;

        if(type == HORIZ) {
            s_center = centre + cv::Point2f(shift, 0);
            if(s_center.x < 0.f) break ;
            else if(s_center.x >= epi.cols) break;
        } else {
            s_center = centre + cv::Point2f(0, shift);
            if(s_center.y < 0.f) break;
            else if(s_center.y >= epi.rows) break;
        }

        double C_i = this->getWeightedColorAndGradScore(s_center, pixel_set, type, epi, epi_grad);
        cv::Vec3f diff = epi.at<cv::Vec3f>(centre) - epi.at<cv::Vec3f>(s_center);
        double w_i = std::exp(-std::abs(shift)/sigma_s - std::abs(cv::norm(diff)/sigma_c));

        C_s += C_i * w_i;

    }

    return C_s;
}

void LEDepthEstimator::getPixelSet(const cv::Point2f &centre, std::vector<cv::Point2f> &pixel_set, epi_type type, const float disparity)
{
    CV_Assert(pixel_set.empty());
    if(type == HORIZ) {
        uint16_t s_hat = this->light_field->s()/2;

        for(uint16_t s = 0; s < this->light_field->s(); s++) {
            if(s == s_hat) continue;

            float u_s = (s - s_hat) * disparity;
            pixel_set.push_back(cv::Point2f(u_s, s));
        }
    } else {
        uint16_t t_hat = this->light_field->t()/2;
        for(uint16_t t = 0; t < this->light_field->t(); t++) {
            if(t == t_hat) continue;

            float v_t = (t_hat - t) * disparity;
            pixel_set.push_back(cv::Point2f(t, v_t));
        }
    }
}

double LEDepthEstimator::getWeightedColorAndGradScore(const cv::Point2f &center, const std::vector<cv::Point2f> pixel_set, epi_type type, const cv::Mat &epi, const cv::Mat &epi_grad)
{
    CV_Assert(epi.type() == CV_32FC3 && epi_grad.type() == CV_32FC3);

    cv::Vec3f c_center = epi.at<cv::Vec3f>(center);
    cv::Vec3f g_center = epi_grad.at<cv::Vec3f>(center);

    int N = 0;
    double SUM = 0.;
    for(const cv::Point2f& point : pixel_set) {
        cv::Vec3f c_inter;
        cv::Vec3f g_inter;
        cv::Point2f s_point;

        if(type == HORIZ) {
            s_point = point + cv::Point2f(center.x, 0);
            if(s_point.x < -0.5f || s_point.x > epi.cols - 0.5f) continue;

            int s = int(s_point.y);
            int u = s_point.x < 0.f ? 0 : int(std::floor(s_point.x));
            double delta = s_point.x < 0.f ? double(-s_point.x) : double(s_point.x - std::floor(s_point.x));

            cv::Vec3f c_b = s_point.x < 0.f ? 0. : epi.at<cv::Vec3f>(s, u+1);
            cv::Vec3f c_a = epi.at<cv::Vec3f>(s, u);

            c_inter = (1. - delta) * c_a + delta * c_b;

            cv::Vec3f gx_b = s_point.x < 0.f ? 0. : epi_grad.at<cv::Vec3f>(s, u+1);
            cv::Vec3f gx_a = epi_grad.at<cv::Vec3f>(s, u);

            g_inter = (1. - delta) * gx_a + delta * gx_b;


        } else {
            s_point = point + cv::Point2f(0, center.y);
            if(s_point.y < -0.5f || s_point.y > epi.rows - 0.5f) continue;

            int t = int(s_point.x);
            int v = s_point.y < 0.f ? 0 : int(std::floor(s_point.y));
            double delta = s_point.y < 0.f ? double(-s_point.y) : double(s_point.y - std::floor(s_point.y));

            cv::Vec3f c_b = s_point.y < 0.f ? 0. : epi.at<cv::Vec3f>(v+1, t);
            cv::Vec3f c_a = epi.at<cv::Vec3f>(v, t);

            c_inter = (1. - delta) * c_a + delta * c_b;

            cv::Vec3f gy_b = s_point.x < 0.f ? 0. : epi_grad.at<cv::Vec3f>(v+1, t);
            cv::Vec3f gy_a = epi_grad.at<cv::Vec3f>(v, t);

            g_inter = (1. - delta) * gy_a + delta * gy_b;

        }

        SUM += std::min(cvutils::normL1(c_inter, c_center), thresh_color) +
               std::min(cvutils::normL1(g_inter, g_center), thresh_gradient);
        N++;

    }

    return SUM/N;
}

void LEDepthEstimator::postProcess()
{
    cv::Mat depth[2];

    for(int epi = HORIZ; epi < END; epi = epi+1) {
        this->computeDisparityFromPlane(epi_type(epi));
        light_field->convertDisparityToDepth(this->current_disp[epi], depth[epi]);
    }

    cv::Mat combo_depth = cv::Mat::zeros(this->height, this->width, CV_32F);
    float thresh = 0.1f;

    for(int v = 0; v < this->height; v++) {
        for(int u = 0; u < this->width; u++) {
            float h_d = depth[HORIZ].at<float>(v, u);
            float v_d = depth[VERT].at<float>(v, u);

            if(std::abs(h_d - v_d) < thresh)
                combo_depth.at<float>(v, u) = std::min(h_d, v_d);
            else
                combo_depth.at<float>(v, u) = 0;
        }
    }

    cv::Mat depth_error = combo_depth - gt_depth;
    cv::Mat depth_error_sq = depth_error.mul(depth_error);
    double mse = cv::sum(depth_error_sq)[0]/(combo_depth.rows * combo_depth.cols);

    cv::Mat depthmap;
    combo_depth.convertTo(depthmap, -1, 1.f, -17.f);
    depthmap *= 50;

    cv::imwrite(save_dir + cv::format("/resultHoriz.png"), depthmap);
    std::cout << cv::format("Final MSE : %15.4f", mse) << std::endl;
}

void LEDepthEstimator::correctMarginLabels()
{
    for(int epi = HORIZ; epi < END; epi = epi+1) {
        cv::Rect inner_rect(2, 2, this->width - 4*M, this->height - 4*M);
        cv::Mat curr_label = this->current_plane_label[epi](inner_rect).clone();
        cv::copyMakeBorder(curr_label, curr_label, 2, 2, 2, 2, cv::BORDER_REPLICATE);
        this->current_plane_label[epi] = curr_label;
    }
}

void LEDepthEstimator::evaluate(int iter)
{
    cv::Mat depthmap[2];
    cv::Mat errormap[2];
    double mse[2];

    for(int epi = HORIZ; epi < END; epi = epi+1) {
        this->computeDisparityFromPlane(epi_type(epi));
        cv::Mat depth;
        light_field->convertDisparityToDepth(current_disp[epi], depth);
        cv::Mat depth_error = depth - gt_depth;
        cv::Mat depth_error_sq = depth_error.mul(depth_error);
        mse[epi] = cv::sum(depth_error_sq)[0]/(depth.rows * depth.cols);

        depth.convertTo(depthmap[epi], -1, 1.f, -17.f);
        depthmap[epi] *= 50;
        errormap[epi] = depth_error_sq;
    }


    cv::imwrite(save_dir + cv::format("/result%dD%02d.png", HORIZ, iter), depthmap[HORIZ]);
    cv::imwrite(save_dir + cv::format("/result%dD%02d.png", VERT, iter), depthmap[VERT]);
    cv::imwrite(save_dir + cv::format("/result%dE%02d.png", HORIZ, iter), errormap[HORIZ] * 255);
    cv::imwrite(save_dir + cv::format("/result%dE%02d.png", VERT, iter), errormap[VERT] * 255);

    std::cout << cv::format("%2d%15.4f%15.4f", iter, mse[HORIZ], mse[VERT]) << std::endl;
}

void LEDepthEstimator::serializeEPIData()
{
    std::ofstream ofs("serialized/epi_data.dat", std::ios::binary);

    int size;
    int cols, rows, type;

    size = this->h_epi_arr.size();
    ofs.write((const char *)&size, sizeof(int));

    cols = this->h_epi_arr[0].cols;
    rows = this->h_epi_arr[0].rows;
    type = this->h_epi_arr[0].type();

    ofs.write((const char *)&cols, sizeof(int));
    ofs.write((const char *)&rows, sizeof(int));
    ofs.write((const char *)&type, sizeof(int));

    unsigned int data_size = rows * cols * this->h_epi_arr[0].elemSize();
    for(const cv::Mat& mat: this->h_epi_arr) {
        ofs.write((const char *)mat.ptr(), data_size);
    }

    size = this->h_epi_grad_arr.size();
    ofs.write((const char *)&size, sizeof(int));

    cols = this->h_epi_grad_arr[0].cols;
    rows = this->h_epi_grad_arr[0].rows;
    type = this->h_epi_grad_arr[0].type();

    ofs.write((const char *)&cols, sizeof(int));
    ofs.write((const char *)&rows, sizeof(int));
    ofs.write((const char *)&type, sizeof(int));

    data_size = rows * cols * this->h_epi_grad_arr[0].elemSize();
    for(const cv::Mat& mat: this->h_epi_grad_arr) {
        ofs.write((const char *)mat.ptr(), data_size);
    }

    size = this->v_epi_arr.size();
    ofs.write((const char *)&size, sizeof(int));

    cols = this->v_epi_arr[0].cols;
    rows = this->v_epi_arr[0].rows;
    type = this->v_epi_arr[0].type();

    ofs.write((const char *)&cols, sizeof(int));
    ofs.write((const char *)&rows, sizeof(int));
    ofs.write((const char *)&type, sizeof(int));

    data_size = rows * cols * this->v_epi_arr[0].elemSize();
    for(const cv::Mat& mat: this->v_epi_arr) {
        ofs.write((const char *)mat.ptr(), data_size);
    }

    size = this->v_epi_grad_arr.size();
    ofs.write((const char *)&size, sizeof(int));

    cols = this->v_epi_grad_arr[0].cols;
    rows = this->v_epi_grad_arr[0].rows;
    type = this->v_epi_grad_arr[0].type();

    ofs.write((const char *)&cols, sizeof(int));
    ofs.write((const char *)&rows, sizeof(int));
    ofs.write((const char *)&type, sizeof(int));

    data_size = rows * cols * this->v_epi_grad_arr[0].elemSize();
    for(const cv::Mat& mat: this->v_epi_grad_arr) {
        ofs.write((const char *)mat.ptr(), data_size);
    }

    ofs.close();
}

void LEDepthEstimator::deserializeEPIData()
{
    std::ifstream ifs("serialized/epi_data.dat", std::ios::binary);

    int s_hat = light_field->s()/2;
    int t_hat = light_field->t()/2;

    int size;
    int cols, rows, type;
    unsigned int data_size;

    ifs.read((char *)&size, sizeof(int));
    this->h_epi_arr.resize(size);

    ifs.read((char *)&cols, sizeof(int));
    ifs.read((char *)&rows, sizeof(int));
    ifs.read((char *)&type, sizeof(int));

    for(cv::Mat& mat: this->h_epi_arr) {
        mat.create(rows, cols, type);
        data_size = rows * cols * mat.elemSize();
        ifs.read((char *)mat.ptr(), data_size);
    }

    ifs.read((char *)&size, sizeof(int));
    this->h_epi_grad_arr.resize(size);

    ifs.read((char *)&cols, sizeof(int));
    ifs.read((char *)&rows, sizeof(int));
    ifs.read((char *)&type, sizeof(int));

    for(cv::Mat& mat: this->h_epi_grad_arr) {
        mat.create(rows, cols, type);
        data_size = rows * cols * mat.elemSize();
        ifs.read((char *)mat.ptr(), data_size);
    }

    ifs.read((char *)&size, sizeof(int));
    this->v_epi_arr.resize(size);
    ifs.read((char *)&cols, sizeof(int));
    ifs.read((char *)&rows, sizeof(int));
    ifs.read((char *)&type, sizeof(int));

    for(cv::Mat& mat: this->v_epi_arr) {
        mat.create(rows, cols, type);
        data_size = rows * cols * mat.elemSize();
        ifs.read((char *)mat.ptr(), data_size);
    }

    ifs.read((char *)&size, sizeof(int));
    this->v_epi_grad_arr.resize(size);

    ifs.read((char *)&cols, sizeof(int));
    ifs.read((char *)&rows, sizeof(int));
    ifs.read((char *)&type, sizeof(int));

    for(cv::Mat& mat: this->v_epi_grad_arr) {
        mat.create(rows, cols, type);
        data_size = rows * cols * mat.elemSize();
        ifs.read((char *)mat.ptr(), data_size);
    }

    ifs.close();
}
