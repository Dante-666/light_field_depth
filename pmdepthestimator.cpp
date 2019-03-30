/**
 * File              : src/depthestimator.cpp
 * Author            : Siddharth J. Singh <j.singh.logan@gmail.com>
 * Date              : 10.08.2017
 * Last Modified Date: 22.10.2018
 * Last Modified By  : Siddharth J. Singh <j.singh.logan@gmail.com>
 */

/**
 * src/depthestimator.cpp
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
#include "pmdepthestimator.h"

PMDepthEstimator::PMDepthEstimator(LfContainer *light_field, float min_disp, float max_disp, std::string save_dir)
{
    this->light_field = light_field;
    this->width = light_field->u();
    this->height = light_field->v();

    this->regions.push_back(Region(this->width-2*M, this->height-2*M, 100));
    this->regions.push_back(Region(this->width-2*M, this->height-2*M, 50));
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
//    if (params.neighborNum >= 8)
//    {
    neighbours[NW] = cv::Point(-1, -1);
    neighbours[NE] = cv::Point(+1, -1);
    neighbours[SW] = cv::Point(-1, +1);
    neighbours[SE] = cv::Point(+1, +1);

    //This generates a vector of coordinates that can be used for inner product
    this->initializeCoordinates();
    //This generates random planes and associates them with each pixel
    this->initializeRandomPlane(HORIZ);
    this->initializeRandomPlane(VERT);

    //One time effort for both EPIs
    this->initializeSmoothnessCoeff();

    //Set the ground truths for central image from the light field data
    this->light_field->getGTDepth(this->light_field->s()/2, this->light_field->t()/2, this->gt_depth);
    this->light_field->convertDepthToDisparity(this->gt_depth, this->gt_disp);

}

PMDepthEstimator::~PMDepthEstimator() {
    //delete allowed_disp;
}

Plane PMDepthEstimator::createRandomLabel(cv::Point s) const
{
    float zs = cv::theRNG().uniform(float(min_disp), float(max_disp));
    float vs = 0.f;

    cv::Vec3d n = cvutils::getRandomUnitVector(CV_PI / 3);
    return Plane::CreatePlane(n, zs, float(s.x), float(s.y), vs);
}

void PMDepthEstimator::initializeCoordinates() {
    for(int j = 0; j < height; j++) {
        for(int i = 0; i < width; i++) {
            coordinates.at<cv::Vec<float, 4> >(j, i) = cv::Vec<float, 4>(float(i), float(j), 1.f, 0.f);
        }
    }
}

void PMDepthEstimator::initializeRandomPlane(enum epi_type type) {
    for(int j = 0; j < height; j++) {
        for(int i = 0; i < width; i++) {
            this->current_plane_label[type].at<Plane>(j, i) = this->createRandomLabel(cv::Point(i, j));
        }
    }
}

void PMDepthEstimator::initializeSmoothnessCoeff()
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
    }
}

void PMDepthEstimator::computeDisparityFromPlane(enum epi_type type) {
    this->current_disp[type] = cvutils::channelDot(this->coordinates, this->current_plane_label[type]);
}

float PMDepthEstimator::getDisparityPerturbationWidth(int iter)
{
    return (this->max_disp - this->min_disp) * pow(0.5f, iter + 1);
}

void PMDepthEstimator::tests() {

//    this->prefetchEPIData(false);

//    std::cout<<v_epi_arr[0].type()<<std::endl;
//    std::cout<<h_epi_arr[0].type()<<std::endl;
//    std::cout<<v_epi_grad_arr[0].type()<<std::endl;
//    std::cout<<h_epi_grad_arr[0].type()<<std::endl;

//    std::cout<<this->getCostFromExtendedSet(cv::Point(4, 739),
//                                 VERT,
//                                 1.25,
//                                 this->v_epi_arr[0],
//                                 this->v_epi_grad_arr[0])<<std::endl;


//    std::cout<<this->getRegionIDfromPixel(cv::Point(1, 1))<<std::endl;
//    std::cout<<this->getRegionIDfromPixel(cv::Point(765, 765))<<std::endl;

}

void PMDepthEstimator::run()
{
    int iter = 0;

    std::chrono::time_point t_0 = std::chrono::high_resolution_clock::now();

    std::cout<<"Prefetching EPI Data..."<<std::endl;
//    this->prefetchEPIData(true);
    this->prefetchEPIData(false);
    std::chrono::time_point t_1 = std::chrono::high_resolution_clock::now();
    std::cout<<"Prefetching EPI Data took : "<<std::chrono::duration_cast<std::chrono::seconds>(t_1 - t_0).count()<<" seconds"<<std::endl;

    while(getDisparityPerturbationWidth(iter) > 0.01) {

        std::cout<<"Algorithm ongoing, iteration : "<<iter<<std::endl;
        std::chrono::time_point t_i = std::chrono::high_resolution_clock::now();

        this->runHorizontalSpatialPropagation(iter);
        std::chrono::time_point t_j = std::chrono::high_resolution_clock::now();
        std::cout<<"Spatial Prop. in horizontal direction : "<<std::chrono::duration_cast<std::chrono::milliseconds>(t_j - t_i).count()<<" milliseconds"<<std::endl;
        t_i = t_j;

        this->runVerticalSpatialPropagation(iter);
        t_j = std::chrono::high_resolution_clock::now();
        std::cout<<"Spatial Prop. in vertical direction : "<<std::chrono::duration_cast<std::chrono::milliseconds>(t_j - t_i).count()<<" milliseconds"<<std::endl;
        t_i = t_j;

        this->perturbatePlaneLabels(iter);
        t_j = std::chrono::high_resolution_clock::now();
        std::cout<<"Perturbation check : "<<std::chrono::duration_cast<std::chrono::milliseconds>(t_j - t_i).count()<<" milliseconds"<<std::endl;

        if(iter >=2) {

                    std::chrono::time_point t_i = std::chrono::high_resolution_clock::now();

                    this->runHorizontalRegionPropagation(iter);
                    std::chrono::time_point t_j = std::chrono::high_resolution_clock::now();
                    std::cout<<"Region Prop. in horizontal direction : "<<std::chrono::duration_cast<std::chrono::milliseconds>(t_j - t_i).count()<<" milliseconds"<<std::endl;
                    t_i = t_j;

            //        this->runVerticalRegionPropagation(iter);
                    t_j = std::chrono::high_resolution_clock::now();
                    std::cout<<"Region Prop. in vertical direction : "<<std::chrono::duration_cast<std::chrono::milliseconds>(t_j - t_i).count()<<" milliseconds"<<std::endl;
                    std::cout<<"=========================================================="<<std::endl;
        }

        this->evaluate(iter++);
    }

//    int gc_iter = 2;

//    while(getDisparityPerturbationWidth(gc_iter) > 0.01) {

//        std::chrono::time_point t_i = std::chrono::high_resolution_clock::now();

//        this->runHorizontalRegionPropagation(gc_iter);
//        std::chrono::time_point t_j = std::chrono::high_resolution_clock::now();
//        std::cout<<"Region Prop. in horizontal direction : "<<std::chrono::duration_cast<std::chrono::milliseconds>(t_j - t_i).count()<<" milliseconds"<<std::endl;
//        t_i = t_j;

////        this->runVerticalRegionPropagation(iter);
//        t_j = std::chrono::high_resolution_clock::now();
//        std::cout<<"Region Prop. in vertical direction : "<<std::chrono::duration_cast<std::chrono::milliseconds>(t_j - t_i).count()<<" milliseconds"<<std::endl;
//        std::cout<<"=========================================================="<<std::endl;
//        this->evaluate(iter + gc_iter++);

//    }

    std::chrono::time_point t_n = std::chrono::high_resolution_clock::now();
    std::cout<<"Terminated the algorithm, total time taken : "<<std::chrono::duration_cast<std::chrono::seconds>(t_n - t_0).count()<<" seconds"<<std::endl;

//    this->computeDisparityFromPlane(HORIZ);
//    this->light_field->convertDisparityToDepth(this->current_disp[HORIZ], this->current_depth[HORIZ]);

//    this->computeDisparityFromPlane(VERT);
//    this->light_field->convertDisparityToDepth(this->current_disp[VERT], this->current_depth[VERT]);

//    cv::Mat w_gt_depth, w_current_depth[2];
//    this->gt_depth.convertTo(w_gt_depth, -1, 1.f, -17.f);
//    w_gt_depth *= 50;
//    cv::imwrite("gt_depth.png", w_gt_depth);

//    this->current_depth[HORIZ].convertTo(w_current_depth[HORIZ], -1, 1.f, -17.f);
//    this->current_depth[VERT].convertTo(w_current_depth[VERT], -1, 1.f, -17.f);
//    w_current_depth[HORIZ] *= 50;
//    w_current_depth[VERT] *= 50;

//    cv::imwrite("current_horiz_depth.png", w_current_depth[HORIZ]);
//    cv::imwrite("current_vert_depth.png", w_current_depth[VERT]);

    //    this->postProcess();
}

inline bool PMDepthEstimator::isValidLabel(Plane label, cv::Point pos)
{
    float ds = label.GetZ(pos);
    float a5 = label.a * 5;
    float b5 = label.b * 5;
    float d;

    return (ds >= this->min_disp && ds <= this->max_disp
            && ((d = ds + a5 + b5) >= this->min_disp) && d <= this->max_disp
            && ((d = ds + a5 - b5) >= this->min_disp) && d <= this->max_disp
            && ((d = ds - a5 + b5) >= this->min_disp) && d <= this->max_disp
            && ((d = ds - a5 - b5) >= this->min_disp) && d <= this->max_disp);
}

void PMDepthEstimator::prefetchEPIData(bool archive)
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

void PMDepthEstimator::runHorizontalSpatialPropagation(int iter) {

    uint16_t s_hat = light_field->s()/2;

    this->computeDisparityFromPlane(HORIZ);

    #pragma omp parallel for
    for(uint16_t v = 0; v < this->height; v++) {
        for(uint16_t u = 0; u < this->width; u++) {
            /*
             * For each pixel p = (u, v) compute the set of allowed Radiances;
             */
            float d_u = this->current_disp[HORIZ].at<float>(v, u);
            cv::Point2f centre(u, s_hat);

            double curr_cost = this->getCostFromExtendedSet(centre, HORIZ, d_u,
                                                            this->h_epi_arr[v],
                                                            this->h_epi_grad_arr[v]);

            this->current_cost[HORIZ].at<float>(v, u) = curr_cost;

            /*
             * Check for right pixel in even iterations and left pixels in the odd iterations
             */
            if(iter % 2 == 0) {
                if(u+1 >= this->width) continue;

                Plane right_plane = this->current_plane_label[HORIZ].at<Plane>(v, u+1);
                cv::Vec<float, 4> curr_coordinate = this->coordinates.at<cv::Vec<float, 4> >(v, u);
                float d_r = curr_coordinate.dot(right_plane.toVec4());
                double neighbour_cost = this->getCostFromExtendedSet(centre, HORIZ, d_r,
                                                                     this->h_epi_arr[v],
                                                                     this->h_epi_grad_arr[v]);

//                if(!isValidLabel(this->current_plane_label[HORIZ].at<Plane>(v, u+1), cv::Point(u+1, v)))
//                    neighbour_cost = INVALID_COST;

                if(neighbour_cost < curr_cost) {
                    this->current_plane_label[HORIZ].at<Plane>(v, u) = right_plane;
                    this->current_cost[HORIZ].at<float>(v, u) = neighbour_cost;
                }

            } else {
                if(u-1 < 0) continue;

                Plane left_plane = this->current_plane_label[HORIZ].at<Plane>(v, u-1);
                cv::Vec<float, 4> curr_coordinate = this->coordinates.at<cv::Vec<float, 4> >(v, u);
                float d_l = curr_coordinate.dot(left_plane.toVec4());
                double neighbour_cost = this->getCostFromExtendedSet(centre, HORIZ, d_l,
                                                                      this->h_epi_arr[v],
                                                                      this->h_epi_grad_arr[v]);

//                if(!isValidLabel(this->current_plane_label[HORIZ].at<Plane>(v, u-1), cv::Point(u-1, v)))
//                    neighbour_cost = INVALID_COST;

                if(neighbour_cost < curr_cost) {
                    this->current_plane_label[HORIZ].at<Plane>(v, u) = left_plane;
                    this->current_cost[HORIZ].at<float>(v, u) = neighbour_cost;
                }
            }

        }
    }

}

void PMDepthEstimator::runVerticalSpatialPropagation(int iter)
{
    uint16_t t_hat = light_field->t()/2;

    this->computeDisparityFromPlane(VERT);

    #pragma omp parallel for
    for(uint16_t u = 0; u < this->width; u++) {
        for(uint16_t v = 0; v < this->height; v++) {
            /**
             * For each pixel p = (u, v) compute the set of allowed Radiances;
             */
            float d_v = this->current_disp[VERT].at<float>(v, u);
            cv::Point2f centre(t_hat, v);

            double curr_cost = this->getCostFromExtendedSet(centre, VERT, d_v,
                                                             this->v_epi_arr[u],
                                                             this->v_epi_grad_arr[u]);

            this->current_cost[VERT].at<float>(v, u) = curr_cost;

            /*
             * Check for bottom pixel in even iterations and top pixels in the odd iterations
             */
            if(iter % 2 == 0) {
                if(v+1 >= this->height) continue;

                Plane bottom_plane = this->current_plane_label[VERT].at<Plane>(v+1, u);
                cv::Vec<float, 4> curr_coordinate = this->coordinates.at<cv::Vec<float, 4> >(v, u);
                float d_b = curr_coordinate.dot(bottom_plane.toVec4());

                double neighbour_cost = this->getCostFromExtendedSet(centre, VERT, d_b,
                                                                      this->v_epi_arr[u],
                                                                      this->v_epi_grad_arr[u]);

                if(neighbour_cost < curr_cost) {
                    this->current_plane_label[VERT].at<Plane>(v, u) = bottom_plane;
                    this->current_cost[VERT].at<float>(v, u) = neighbour_cost;
                }

            } else {
                if(v-1 < 0) continue;

                Plane top_plane = this->current_plane_label[VERT].at<Plane>(v-1, u);
                cv::Vec<float, 4> curr_coordinate = this->coordinates.at<cv::Vec<float, 4> >(v, u);
                float d_t = curr_coordinate.dot(top_plane.toVec4());

                double neighbour_cost = this->getCostFromExtendedSet(centre, VERT, d_t,
                                                                     this->v_epi_arr[u],
                                                                     this->v_epi_grad_arr[u]);

                if(neighbour_cost < curr_cost) {
                    this->current_plane_label[VERT].at<Plane>(v, u) = top_plane;
                    this->current_cost[VERT].at<float>(v, u) = neighbour_cost;
                }
            }

        }
    }

}

void PMDepthEstimator::perturbatePlaneLabels(int iter)
{

    cv::Mat perturb_plane_label[2];
    perturb_plane_label[HORIZ] = cv::Mat::zeros(this->height, this->width, cv::DataType<Plane>::type);
    perturb_plane_label[VERT] = cv::Mat::zeros(this->height, this->width, cv::DataType<Plane>::type);

    #pragma omp parallel for
    for(uint16_t v = 0; v < this->height; v++) {
        for(uint16_t u = 0; u < this->width; u++) {
            for(uint16_t type = HORIZ; type < END; type++) {
                Plane curr_plane = this->current_plane_label[type].at<Plane>(v, u);

                float zs = curr_plane.GetZ(u, v);
                float dz = this->getDisparityPerturbationWidth(iter);
                float minz = std::max(this->min_disp, zs - dz);
                float maxz = std::min(this->max_disp, zs + dz);
                zs = cv::theRNG().uniform(minz, maxz);

                cv::Vec3f dn = cvutils::getRandomUnitVector() * pow(0.5f, iter);
                cv::Vec3f n = curr_plane.GetNormal() + dn;
                n = n/sqrt(n.ddot(n));

                perturb_plane_label[type].at<Plane>(v, u) = Plane::CreatePlane(n, zs, u, v, 0.f);
            }
        }
    }

    uint16_t s_hat = light_field->s()/2;
    uint16_t t_hat = light_field->t()/2;

    cv::Mat perturb_disp[2];
    perturb_disp[HORIZ] = cvutils::channelDot(this->coordinates, perturb_plane_label[HORIZ]);
    perturb_disp[VERT] = cvutils::channelDot(this->coordinates, perturb_plane_label[VERT]);

    #pragma omp parallel for
    for(uint16_t v = 0; v < this->height; v++) {
        for(uint16_t u = 0; u < this->width; u++) {

            float d_u = perturb_disp[HORIZ].at<float>(v, u);
            cv::Point2f centre(u, s_hat);

            double curr_cost = this->current_cost[HORIZ].at<float>(v, u);
            double perturbed_cost = this->getCostFromExtendedSet(centre, HORIZ, d_u,
                                                                  this->h_epi_arr[v],
                                                                  this->h_epi_grad_arr[v]);

            if(perturbed_cost < curr_cost) {
                this->current_plane_label[HORIZ].at<Plane>(v, u) = perturb_plane_label[HORIZ].at<Plane>(v, u);
                this->current_cost[HORIZ].at<float>(v, u) = perturbed_cost;
            }
        }
    }

    #pragma omp parallel for
    for(uint16_t u = 0; u < this->width; u++) {
        for(uint16_t v = 0; v < this->height; v++) {

            float d_v = perturb_disp[VERT].at<float>(v, u);
            cv::Point2f centre(t_hat, v);

            double curr_cost = this->current_cost[VERT].at<float>(v, u);
            double perturbed_cost = this->getCostFromExtendedSet(centre, VERT, d_v,
                                                                  this->v_epi_arr[u],
                                                                  this->v_epi_grad_arr[u]);

            if(perturbed_cost < curr_cost) {
                this->current_plane_label[VERT].at<Plane>(v, u) = perturb_plane_label[VERT].at<Plane>(v, u);
                this->current_cost[VERT].at<float>(v, u) = perturbed_cost;
            }
        }
    }

}

void PMDepthEstimator::runHorizontalRegionPropagation(int iter)
{
    cv::Point offset(M, M);
    this->computeDisparityFromPlane(HORIZ);

    for(int r = 0; r < regions.size(); r++) {
        std::vector<Plane> ransac_proposals;
        ransac_proposals.reserve(regions[r].unitRegions.size());

        for(int id = 0; id < regions[r].unitRegions.size(); id++){
            ransac_proposals[id] = this->getRANSACPlane(regions[r].unitRegions[id] + offset, HORIZ);
        }

        cv::Rect inner_rect(M, M, width-2*M, height-2*M);
        cv::Mat proposal_cost = cv::Mat::zeros(this->height, this->width, CV_32F);

        uint16_t s_hat = light_field->s()/2;

        #pragma omp parallel for
        for(uint16_t v = M; v < this->height - 2*M; v++) {
            for(uint16_t u = M; u < this->width - 2*M; u++) {
                cv::Point pt(u, v);
                int id = this->getRegionIDfromPixel(pt, r);
                float d_u = ransac_proposals[id].GetZ(pt);

                cv::Point centre(u, s_hat);
                double proposed_cost = this->getCostFromExtendedSet(centre, HORIZ, d_u,
                                                                    this->h_epi_arr[v],
                                                                    this->h_epi_grad_arr[v]);
                proposal_cost.at<float>(pt) = proposed_cost;
            }
        }

//        #pragma omp parallel for
        for(int id = 0; id < regions[r].unitRegions.size(); id++) {

            std::vector<cv::Mat> cost_fp_fq(neighbours.size());
            std::vector<cv::Mat> cost_fp_alpha(neighbours.size());
            std::vector<cv::Mat> cost_alpha_fq(neighbours.size());

            cv::Rect unitRegion = regions[r].unitRegions[id];

            Plane alpha = ransac_proposals[id];

            cv::Mat fp = this->current_plane_label[HORIZ](unitRegion + offset);
            cv::Mat coord_p = this->coordinates(unitRegion + offset);

            cv::Mat dp_fp = cvutils::channelDot(fp, coord_p);
            cv::Mat dp_alpha = cvutils::channelSum(coord_p.mul(alpha.toScalar()));

            for(int i = 0; i < neighbours.size(); i++) {

                cv::Rect neighbourUnitRegion = unitRegion + neighbours[i];

                cv::Mat fq = this->current_plane_label[HORIZ](neighbourUnitRegion + offset);
                cv::Mat coord_q = this->coordinates(neighbourUnitRegion + offset);

                cv::Mat dq_fq = cvutils::channelDot(fq, coord_q);
                cv::Mat dq_alpha = cvutils::channelSum(coord_q.mul(alpha.toScalar()));

                cv::Mat dp_fq = cvutils::channelDot(fq, coord_p);
                cv::Mat dq_fp = cvutils::channelDot(fp, coord_q);

                cv::Mat smoothness_term = this->smoothness_coeff[i](unitRegion);

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

            int n = unitRegion.width * unitRegion.height;
            typedef Graph<float, float, double> G;
            G graph(n, 4 * n);

            graph.add_node(n);
            cv::Point unitRegion_origin(unitRegion.x, unitRegion.y);

            for(int y = 0; y < unitRegion.height; y++) {
                for(int x = 0; x < unitRegion.width; x++) {
                    int i = y*unitRegion.width + x;
                    cv::Point unitRegion_coord(x, y);
                    cv::Point imageRegion_coord = unitRegion_coord + unitRegion_origin + offset;
                    graph.add_tweights(i, this->current_cost[HORIZ].at<float>(imageRegion_coord), proposal_cost.at<float>(imageRegion_coord));

                    bool x0 = x == 0;
                    bool x1 = x == unitRegion.width - 1;
                    bool y0 = y == 0;
                    bool y1 = y == unitRegion.height - 1;

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
            for(int y = 0; y < unitRegion.height; y++) {
                for(int x = 0; x < unitRegion.width - 1; x++) {
                    int i = y*unitRegion.width + x;
                    int j = y*unitRegion.width + x + 1;

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
            for(int y = 0; y < unitRegion.height - 1; y++) {
                for(int x = 0; x < unitRegion.width; x++) {
                    int i = y*unitRegion.width + x;
                    int j = (y + 1)*unitRegion.width + x;

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
            for(int y = 0; y < unitRegion.height - 1; y++) {
                for(int x = 0; x < unitRegion.width; x++) {
                    int i = y*unitRegion.width + x;
                    int j = (y + 1)*unitRegion.width + x - 1;

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
            for(int y = 0; y < unitRegion.height - 1; y++) {
                for(int x = 0; x < unitRegion.width - 1; x++) {
                    int i = y*unitRegion.width + x;
                    int j = (y + 1)*unitRegion.width + x + 1;

                    float B = cost_alpha_fq[SW].at<float>(y, x);
                    float C = cost_fp_alpha[SW].at<float>(y, x);
                    float D = cost_fp_fq[SW].at<float>(y, x);

                    graph.add_edge(i, j, std::max(0.f, B + C - D), 0);
                    graph.add_tweights(i, C, 0);
                    graph.add_tweights(j, D - C, 0);

                }
            }

            double flow = graph.maxflow();

            for (int y = 0; y < unitRegion.height; y++){
                for (int x = 0; x < unitRegion.width; x++){
                    if(graph.what_segment(y*unitRegion.width + x) == G::SOURCE) {
                        cv::Point unitRegion_coord(x, y);
                        cv::Point imageRegion_coord = unitRegion_coord + unitRegion_origin + offset;

                        this->current_cost[HORIZ].at<float>(imageRegion_coord) = proposal_cost.at<float>(imageRegion_coord);
                        this->current_plane_label[HORIZ].at<Plane>(imageRegion_coord) = alpha;
                    }
                }
            }

        }

        std::vector<Plane> random_proposals;
        random_proposals.reserve(regions[r].unitRegions.size());

        for(int id = 0; id < regions[r].unitRegions.size(); id++){
            random_proposals[id] = this->getRandomPlane(regions[r].unitRegions[id] + offset, HORIZ, iter);
        }

        #pragma omp parallel for
        for(uint16_t v = M; v < this->height - 2*M; v++) {
            for(uint16_t u = M; u < this->width - 2*M; u++) {
                cv::Point pt(u, v);
                int id = this->getRegionIDfromPixel(pt, r);
                float d_u = random_proposals[id].GetZ(pt);

                cv::Point centre(u, s_hat);
                double proposed_cost = this->getCostFromExtendedSet(centre, HORIZ, d_u,
                                                                    this->h_epi_arr[v],
                                                                    this->h_epi_grad_arr[v]);
                proposal_cost.at<float>(pt) = proposed_cost;
            }
        }

        #pragma omp parallel for
        for(int id = 0; id < regions[r].unitRegions.size(); id++) {

            std::vector<cv::Mat> cost_fp_fq(neighbours.size());
            std::vector<cv::Mat> cost_fp_alpha(neighbours.size());
            std::vector<cv::Mat> cost_alpha_fq(neighbours.size());

            cv::Rect unitRegion = regions[r].unitRegions[id];

            Plane alpha = random_proposals[id];

            cv::Mat fp = this->current_plane_label[HORIZ](unitRegion + offset);
            cv::Mat coord_p = this->coordinates(unitRegion + offset);

            cv::Mat dp_fp = cvutils::channelDot(fp, coord_p);
            cv::Mat dp_alpha = cvutils::channelSum(coord_p.mul(alpha.toScalar()));

            for(int i = 0; i < neighbours.size(); i++) {

                cv::Rect neighbourUnitRegion = unitRegion + neighbours[i];

                cv::Mat fq = this->current_plane_label[HORIZ](neighbourUnitRegion + offset);
                cv::Mat coord_q = this->coordinates(neighbourUnitRegion + offset);

                cv::Mat dq_fq = cvutils::channelDot(fq, coord_q);
                cv::Mat dq_alpha = cvutils::channelSum(coord_q.mul(alpha.toScalar()));

                cv::Mat dp_fq = cvutils::channelDot(fq, coord_p);
                cv::Mat dq_fp = cvutils::channelDot(fp, coord_q);

                cv::Mat smoothness_term = this->smoothness_coeff[i](unitRegion);

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

            int n = unitRegion.width * unitRegion.height;
            typedef Graph<float, float, double> G;
            G graph(n, 4 * n);

            graph.add_node(n);
            cv::Point unitRegion_origin(unitRegion.x, unitRegion.y);

            for(int y = 0; y < unitRegion.height; y++) {
                for(int x = 0; x < unitRegion.width; x++) {
                    int i = y*unitRegion.width + x;
                    cv::Point unitRegion_coord(x, y);
                    cv::Point imageRegion_coord = unitRegion_coord + unitRegion_origin + offset;
                    graph.add_tweights(i, this->current_cost[HORIZ].at<float>(imageRegion_coord), proposal_cost.at<float>(imageRegion_coord));

                    bool x0 = x == 0;
                    bool x1 = x == unitRegion.width - 1;
                    bool y0 = y == 0;
                    bool y1 = y == unitRegion.height - 1;

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
            for(int y = 0; y < unitRegion.height; y++) {
                for(int x = 0; x < unitRegion.width - 1; x++) {
                    int i = y*unitRegion.width + x;
                    int j = y*unitRegion.width + x + 1;

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
            for(int y = 0; y < unitRegion.height - 1; y++) {
                for(int x = 0; x < unitRegion.width; x++) {
                    int i = y*unitRegion.width + x;
                    int j = (y + 1)*unitRegion.width + x;

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
            for(int y = 0; y < unitRegion.height - 1; y++) {
                for(int x = 0; x < unitRegion.width; x++) {
                    int i = y*unitRegion.width + x;
                    int j = (y + 1)*unitRegion.width + x - 1;

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
            for(int y = 0; y < unitRegion.height - 1; y++) {
                for(int x = 0; x < unitRegion.width - 1; x++) {
                    int i = y*unitRegion.width + x;
                    int j = (y + 1)*unitRegion.width + x + 1;

                    float B = cost_alpha_fq[SW].at<float>(y, x);
                    float C = cost_fp_alpha[SW].at<float>(y, x);
                    float D = cost_fp_fq[SW].at<float>(y, x);

                    graph.add_edge(i, j, std::max(0.f, B + C - D), 0);
                    graph.add_tweights(i, C, 0);
                    graph.add_tweights(j, D - C, 0);

                }
            }

            double flow = graph.maxflow();

            for (int y = 0; y < unitRegion.height; y++){
                for (int x = 0; x < unitRegion.width; x++){
                    if(graph.what_segment(y*unitRegion.width + x) == G::SOURCE) {
                        cv::Point unitRegion_coord(x, y);
                        cv::Point imageRegion_coord = unitRegion_coord + unitRegion_origin + offset;

                        this->current_cost[HORIZ].at<float>(imageRegion_coord) = proposal_cost.at<float>(imageRegion_coord);
                        this->current_plane_label[HORIZ].at<Plane>(imageRegion_coord) = alpha;
                    }
                }
            }

        }
    }
}

void PMDepthEstimator::runVerticalRegionPropagation(int iter)
{
    cv::Point offset(M, M);

    this->computeDisparityFromPlane(VERT);
    std::vector<Plane> proposals;
    proposals.reserve(regions[0].unitRegions.size());

    for(int id = 0; id < regions[0].unitRegions.size(); id++){
        proposals[id] = this->getRANSACPlane(regions[0].unitRegions[id] + offset, VERT);
    }

    cv::Rect inner_rect(M, M, width-2*M, height-2*M);
    cv::Mat proposal_cost = cv::Mat::zeros(this->height, this->width, CV_32F);

    uint16_t t_hat = light_field->t()/2;

    #pragma omp parallel for
    for(uint16_t u = M; u < this->width - 2*M; u++) {
        for(uint16_t v = M; v < this->height - 2*M; v++) {
            cv::Point pt(u, v);
            int id = this->getRegionIDfromPixel(pt, 0);
            float d_u = proposals[id].GetZ(pt);

            cv::Point centre(t_hat, v);
            double proposed_cost = this->getCostFromExtendedSet(centre, VERT, d_u,
                                                                this->v_epi_arr[u],
                                                                this->v_epi_grad_arr[u]);
                proposal_cost.at<float>(pt) = proposed_cost;
        }
    }

    #pragma omp parallel for
    for(int id = 0; id < regions[0].unitRegions.size(); id++) {

        std::vector<cv::Mat> cost_fp_fq(neighbours.size());
        std::vector<cv::Mat> cost_fp_alpha(neighbours.size());
        std::vector<cv::Mat> cost_alpha_fq(neighbours.size());

        cv::Rect unitRegion = regions[0].unitRegions[id];

        Plane alpha = proposals[id];

        cv::Mat fp = this->current_plane_label[VERT](unitRegion + offset);
        cv::Mat coord_p = this->coordinates(unitRegion + offset);

        cv::Mat dp_fp = cvutils::channelDot(fp, coord_p);
        cv::Mat dp_alpha = cvutils::channelSum(coord_p.mul(alpha.toScalar()));

        for(int i = 0; i < neighbours.size(); i++) {

            cv::Rect neighbourUnitRegion = unitRegion + neighbours[i];

            cv::Mat fq = this->current_plane_label[VERT](neighbourUnitRegion + offset);
            cv::Mat coord_q = this->coordinates(neighbourUnitRegion + offset);

            cv::Mat dq_fq = cvutils::channelDot(fq, coord_q);
            cv::Mat dq_alpha = cvutils::channelSum(coord_q.mul(alpha.toScalar()));

            cv::Mat dp_fq = cvutils::channelDot(fq, coord_p);
            cv::Mat dq_fp = cvutils::channelDot(fp, coord_q);

            cv::Mat smoothness_term = this->smoothness_coeff[i](unitRegion);

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

        int n = unitRegion.width * unitRegion.height;
        typedef Graph<float, float, double> G;
        G graph(n, 4 * n);

        graph.add_node(n);
        cv::Point unitRegion_origin(unitRegion.x, unitRegion.y);

        for(int y = 0; y < unitRegion.height; y++) {
            for(int x = 0; x < unitRegion.width; x++) {
                int i = y*unitRegion.width + x;
                cv::Point unitRegion_coord(x, y);
                cv::Point imageRegion_coord = unitRegion_coord + unitRegion_origin + offset;
                graph.add_tweights(i, this->current_cost[VERT].at<float>(imageRegion_coord), proposal_cost.at<float>(imageRegion_coord));

                bool x0 = x == 0;
                bool x1 = x == unitRegion.width - 1;
                bool y0 = y == 0;
                bool y1 = y == unitRegion.height - 1;

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
        for(int y = 0; y < unitRegion.height; y++) {
            for(int x = 0; x < unitRegion.width - 1; x++) {
                int i = y*unitRegion.width + x;
                int j = y*unitRegion.width + x + 1;

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
        for(int y = 0; y < unitRegion.height - 1; y++) {
            for(int x = 0; x < unitRegion.width; x++) {
                int i = y*unitRegion.width + x;
                int j = (y + 1)*unitRegion.width + x;

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
        for(int y = 0; y < unitRegion.height - 1; y++) {
            for(int x = 0; x < unitRegion.width; x++) {
                int i = y*unitRegion.width + x;
                int j = (y + 1)*unitRegion.width + x - 1;

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
        for(int y = 0; y < unitRegion.height - 1; y++) {
            for(int x = 0; x < unitRegion.width - 1; x++) {
                int i = y*unitRegion.width + x;
                int j = (y + 1)*unitRegion.width + x + 1;

                float B = cost_alpha_fq[SW].at<float>(y, x);
                float C = cost_fp_alpha[SW].at<float>(y, x);
                float D = cost_fp_fq[SW].at<float>(y, x);

                graph.add_edge(i, j, std::max(0.f, B + C - D), 0);
                graph.add_tweights(i, C, 0);
                graph.add_tweights(j, D - C, 0);

            }
        }

        double flow = graph.maxflow();

        for (int y = 0; y < unitRegion.height; y++){
            for (int x = 0; x < unitRegion.width; x++){
                if(graph.what_segment(y*unitRegion.width + x) == G::SOURCE) {
                    cv::Point unitRegion_coord(x, y);
                    cv::Point imageRegion_coord = unitRegion_coord + unitRegion_origin + offset;

                    this->current_cost[VERT].at<float>(imageRegion_coord) = proposal_cost.at<float>(imageRegion_coord);
                    this->current_plane_label[VERT].at<Plane>(imageRegion_coord) = alpha;
                }
            }
        }

    }

}

Plane PMDepthEstimator::getExpansionPlane(cv::Rect unitRegion, epi_type type)
{
    int n = cv::theRNG().uniform(0, unitRegion.height * unitRegion.width);

    int xx = n % unitRegion.width;
    int yy = n / unitRegion.width;

    cv::Point pt(unitRegion.x + xx, unitRegion.y + yy);

    return this->current_plane_label[type].at<Plane>(pt);
}

Plane PMDepthEstimator::getRANSACPlane(cv::Rect unitRegion, epi_type type)
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

Plane PMDepthEstimator::getRandomPlane(cv::Rect unitRegion, epi_type type, int iter)
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

inline int PMDepthEstimator::computeSampleCount(int ni, int ptNum, int pf, double conf)
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

inline int PMDepthEstimator::getRegionIDfromPixel(cv::Point pt, int region_id)
{
    cv::Point offset(M, M);
    int regionUnitSize = this->regions[region_id].regionUnitSize;

    int x = (pt.x - offset.x) / regionUnitSize;
    int y = (pt.y - offset.y) / regionUnitSize;

    return x + y * this->regions[region_id].widthBlocks;
}

double PMDepthEstimator::getCostFromExtendedSet(const cv::Point2f &center,
                                                epi_type type,
                                                const float disparity,
                                                const cv::Mat &epi,
                                                const cv::Mat &epi_grad)
{
    if(std::isnan(disparity)) {
        return INVALID_COST;
    }
    std::vector<cv::Point2f> pixel_set;
    this->getPixelSet(center, pixel_set, type, disparity);
    double C_s = 0.;
    double sigma_s = 1, sigma_c = 5;

    for(int shift = -W; shift <= W; shift++) {

        cv::Point2f s_center;

        if(type == HORIZ) {
            s_center = center + cv::Point2f(shift, 0);
            if(s_center.x < 0.f) break ;
            else if(s_center.x >= epi.cols) break;
        } else {
            s_center = center + cv::Point2f(0, shift);
            if(s_center.y < 0.f) break;
            else if(s_center.y >= epi.rows) break;
        }

        double C_i = this->getWeightedColorAndGradScore(s_center, pixel_set, type, epi, epi_grad);
        cv::Vec3f diff = epi.at<cv::Vec3f>(center) - epi.at<cv::Vec3f>(s_center);
        double w_i = std::exp(-std::abs(shift)/sigma_s - std::abs(cv::norm(diff)/sigma_c));

        C_s += C_i * w_i;

    }

    return C_s;

}

inline void PMDepthEstimator::getPixelSet(const cv::Point2f &center,
                                   std::vector<cv::Point2f> &pixel_set,
                                   epi_type type,
                                   float disparity)
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

inline double PMDepthEstimator::getWeightedColorAndGradScore(const cv::Point2f &center,
                                                      const std::vector<cv::Point2f> pixel_set,
                                                      epi_type type,
                                                      const cv::Mat &epi,
                                                      const cv::Mat &epi_grad)
{
    CV_Assert(epi.type() == CV_32FC3 && epi_grad.type() == CV_32FC3);
//    double tau_c = 1000000.0;
//    double tau_g = 5000000.0;

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

void PMDepthEstimator::postProcess()
{
    uint16_t s_hat = this->light_field->s()/2;
    uint16_t t_hat = this->light_field->t()/2;

    cv::Mat central_image;
    this->light_field->getImage(s_hat, t_hat, central_image);
    cv::Mat central_image_blur;

    cv::GaussianBlur(central_image, central_image_blur, cv::Size(3, 3), 0);
    cv::cvtColor(central_image_blur, central_image_blur, cv::COLOR_RGB2Lab);

    //cv::Ptr<cv::ximgproc::SuperpixelLSC> ptr = cv::ximgproc::createSuperpixelLSC(central_image, 10, 0.075);
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> ptr = cv::ximgproc::createSuperpixelSLIC(central_image_blur, cv::ximgproc::SLICO, 10, 10.0f);
    ptr->iterate();
    std::cout<<ptr->getNumberOfSuperpixels()<<std::endl;

    cv::Mat border;
    ptr->getLabelContourMask(border);
    border.convertTo(border, CV_8U);
    cv::Mat invert_border;
    cv::bitwise_not(border, invert_border);
//    cv::imshow("win", border);
//    cv::waitKey();
//    cv::imshow("win", invert_border);
//    cv::waitKey();

    cv::Mat w_gt_depth;
    this->gt_depth.convertTo(w_gt_depth, -1, 1.f, -17.f);
    w_gt_depth *= 50;
    cv::Mat w_gt_depth_wo_border;
    w_gt_depth.copyTo(w_gt_depth_wo_border, invert_border);
    cv::Mat white_all(w_gt_depth.rows, w_gt_depth.cols, w_gt_depth.type(), 255.f);
    cv::Mat white_border;
    white_all.copyTo(white_border, border);
//    cv::imshow("win", w_gt_depth_wo_border);
//    cv::waitKey();
//    cv::imshow("win", white_border);
//    cv::waitKey();

    cv::imwrite("gt_depth.png", w_gt_depth_wo_border);

//    cv::Mat view_image;
//    central_image.copyTo(view_image, invert_border);
//    cv::Mat red_border(view_image.rows, view_image.cols, view_image.type(), cv::Scalar(255, 0, 0));
//    red_border.copyTo(red_border, border);
//    view_image += red_border;
//    cv::imwrite("test_superpixel.png", view_image);

}

double PMDepthEstimator::getMSEdepth()
{
//    cv::Mat error = this->current_depth - this->gt_depth;
//    return cv::mean(error.mul(error))[0];
    return 0.0;
}

double PMDepthEstimator::getMSEdisparity()
{
//    cv::Mat error = this->current_disp - this->gt_disp;
//    return cv::mean(error.mul(error))[0];
    return 0.0;
}

void PMDepthEstimator::evaluate(int iter)
{
//    cv::Mat labeling = labeling_m(energy2.getRectWithoutMargin());
//    double sc2 = energy2.computeSmoothnessCost(labeling_m);
//    double dc2 = cv::sum(unaryCost2)[0];
//    double eng2 = sc2 + dc2;
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
//    cv::imwrite(saveDir + cv::format("/result%dN%02d.png", HORIZ, iter), normalmap * 255);
    cv::imwrite(save_dir + cv::format("/result%dE%02d.png", HORIZ, iter), errormap[HORIZ] * 255);
    cv::imwrite(save_dir + cv::format("/result%dE%02d.png", VERT, iter), errormap[VERT] * 255);

//    std::cout << cv::format("%2d%10.2lf%25.2lf%25.2lf%25.2lf%7.2lf", index, getCurrentTime(), eng2, dc2, sc2, mse) << std::endl;
    std::cout << cv::format("%2d%15.4f%15.4f", iter, mse[HORIZ], mse[VERT]) << std::endl;
}

void PMDepthEstimator::render(const cv::Mat& depth, const cv::Mat& curr_estimate) {
    Gnuplot gp;

    std::vector<float> v_depth(size_t(depth.cols));
    std::vector<float> v_curr_estimate(size_t(curr_estimate.cols));

    for(int i = 0; i < depth.cols; i++) {
        v_depth[size_t(i)] = depth.at<float>(i);
        v_curr_estimate[size_t(i)] = curr_estimate.at<float>(i);
    }

    gp << "set xrange[0:" << depth.cols - 1 << "]\n";
    gp << "plot '-' with points ls 6 lc 'blue' title 'depth', '-' with points ls 4 lc 'red' title 'current\\_estimate'\n";
    gp.send1d(v_depth);
    gp.send1d(v_curr_estimate);

    char c;
    std::cout << "Press any key to continue...(*)"<<std::endl;
    std::cin >> c;

}

void PMDepthEstimator::write_epi_with_grad(const cv::Mat& epi, const cv::Mat& epi_xgrad, const cv::Mat& epi_ygrad, epi_type type) {

    std::vector<cv::Mat> epis;
    epis.push_back(epi);
    epis.push_back(epi_xgrad);
    epis.push_back(epi_ygrad);

    cv::Mat epi_out;
    if(type == HORIZ) {
        cv::vconcat(epis, epi_out);
        cv::imwrite("epi_out_horiz.png", epi_out);
    }
    else {
        cv::hconcat(epis, epi_out);
        cv::imwrite("epi_out_vert.png", epi_out);
    }
}

void PMDepthEstimator::serializeEPIData()
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

void PMDepthEstimator::deserializeEPIData()
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

/*
void PMDepthEstimator::serializeEPIData()
{
    std::ofstream ofs("serialized/epi_data.dat", std::ios::binary);

    int size = this->horiz_epi_arr.size();
    ofs.write((const char *)&size, sizeof(int));

    int cols, rows, type;
    cols = this->horiz_epi_arr[0].cols;
    rows = this->horiz_epi_arr[0].rows;
    type = this->horiz_epi_arr[0].type();

    ofs.write((const char *)&cols, sizeof(int));
    ofs.write((const char *)&rows, sizeof(int));
    ofs.write((const char *)&type, sizeof(int));

    unsigned int data_size = rows * cols * this->horiz_epi_arr[0].elemSize();
    for(const cv::Mat& mat: this->horiz_epi_arr) {
        ofs.write((const char *)mat.ptr(), data_size);
    }

    size = this->horiz_epi_xgrad_arr.size();
    ofs.write((const char *)&size, sizeof(int));

    cols = this->horiz_epi_xgrad_arr[0].cols;
    rows = this->horiz_epi_xgrad_arr[0].rows;
    type = this->horiz_epi_xgrad_arr[0].type();

    ofs.write((const char *)&cols, sizeof(int));
    ofs.write((const char *)&rows, sizeof(int));
    ofs.write((const char *)&type, sizeof(int));

    data_size = rows * cols * this->horiz_epi_xgrad_arr[0].elemSize();
    for(const cv::Mat& mat: this->horiz_epi_xgrad_arr) {
        ofs.write((const char *)mat.ptr(), data_size);
    }

    size = this->horiz_epi_ygrad_arr.size();
    ofs.write((const char *)&size, sizeof(int));

    cols = this->horiz_epi_ygrad_arr[0].cols;
    rows = this->horiz_epi_ygrad_arr[0].rows;
    type = this->horiz_epi_ygrad_arr[0].type();

    ofs.write((const char *)&cols, sizeof(int));
    ofs.write((const char *)&rows, sizeof(int));
    ofs.write((const char *)&type, sizeof(int));

    data_size = rows * cols * this->horiz_epi_ygrad_arr[0].elemSize();
    for(const cv::Mat& mat: this->horiz_epi_ygrad_arr) {
        ofs.write((const char *)mat.ptr(), data_size);
    }

    size = this->vert_epi_arr.size();
    ofs.write((const char *)&size, sizeof(int));

    cols = this->vert_epi_arr[0].cols;
    rows = this->vert_epi_arr[0].rows;
    type = this->vert_epi_arr[0].type();

    ofs.write((const char *)&cols, sizeof(int));
    ofs.write((const char *)&rows, sizeof(int));
    ofs.write((const char *)&type, sizeof(int));

    data_size = rows * cols * this->vert_epi_arr[0].elemSize();
    for(const cv::Mat& mat: this->vert_epi_arr) {
        ofs.write((const char *)mat.ptr(), data_size);
    }

    size = this->vert_epi_xgrad_arr.size();
    ofs.write((const char *)&size, sizeof(int));

    cols = this->vert_epi_xgrad_arr[0].cols;
    rows = this->vert_epi_xgrad_arr[0].rows;
    type = this->vert_epi_xgrad_arr[0].type();

    ofs.write((const char *)&cols, sizeof(int));
    ofs.write((const char *)&rows, sizeof(int));
    ofs.write((const char *)&type, sizeof(int));

    data_size = rows * cols * this->vert_epi_xgrad_arr[0].elemSize();
    for(const cv::Mat& mat: this->vert_epi_xgrad_arr) {
        ofs.write((const char *)mat.ptr(), data_size);
    }

    size = this->vert_epi_ygrad_arr.size();
    ofs.write((const char *)&size, sizeof(int));

    cols = this->vert_epi_ygrad_arr[0].cols;
    rows = this->vert_epi_ygrad_arr[0].rows;
    type = this->vert_epi_ygrad_arr[0].type();

    ofs.write((const char *)&cols, sizeof(int));
    ofs.write((const char *)&rows, sizeof(int));
    ofs.write((const char *)&type, sizeof(int));

    data_size = rows * cols * this->vert_epi_ygrad_arr[0].elemSize();
    for(const cv::Mat& mat: this->vert_epi_ygrad_arr) {
        ofs.write((const char *)mat.ptr(), data_size);
    }

    ofs.close();

}

void PMDepthEstimator::deserializeEPIData()
{
    std::ifstream ifs("serialized/epi_data.dat", std::ios::binary);

    int size;
    ifs.read((char *)&size, sizeof(int));
    this->horiz_epi_arr.resize(size);

    int cols, rows, type;
    ifs.read((char *)&cols, sizeof(int));
    ifs.read((char *)&rows, sizeof(int));
    ifs.read((char *)&type, sizeof(int));

    unsigned int data_size;
    for(cv::Mat& mat: this->horiz_epi_arr) {
        mat.create(rows, cols, type);
        data_size = rows * cols * mat.elemSize();
        ifs.read((char *)mat.ptr(), data_size);
    }

    ifs.read((char *)&size, sizeof(int));
    this->horiz_epi_xgrad_arr.resize(size);

    ifs.read((char *)&cols, sizeof(int));
    ifs.read((char *)&rows, sizeof(int));
    ifs.read((char *)&type, sizeof(int));

    for(cv::Mat& mat: this->horiz_epi_xgrad_arr) {
        mat.create(rows, cols, type);
        data_size = rows * cols * mat.elemSize();
        ifs.read((char *)mat.ptr(), data_size);
    }

    ifs.read((char *)&size, sizeof(int));
    this->horiz_epi_ygrad_arr.resize(size);

    ifs.read((char *)&cols, sizeof(int));
    ifs.read((char *)&rows, sizeof(int));
    ifs.read((char *)&type, sizeof(int));

    for(cv::Mat& mat: this->horiz_epi_ygrad_arr) {
        mat.create(rows, cols, type);
        data_size = rows * cols * mat.elemSize();
        ifs.read((char *)mat.ptr(), data_size);
    }

    ifs.read((char *)&size, sizeof(int));
    this->vert_epi_arr.resize(size);

    ifs.read((char *)&cols, sizeof(int));
    ifs.read((char *)&rows, sizeof(int));
    ifs.read((char *)&type, sizeof(int));

    for(cv::Mat& mat: this->vert_epi_arr) {
        mat.create(rows, cols, type);
        data_size = rows * cols * mat.elemSize();
        ifs.read((char *)mat.ptr(), data_size);
    }

    ifs.read((char *)&size, sizeof(int));
    this->vert_epi_xgrad_arr.resize(size);

    ifs.read((char *)&cols, sizeof(int));
    ifs.read((char *)&rows, sizeof(int));
    ifs.read((char *)&type, sizeof(int));

    for(cv::Mat& mat: this->vert_epi_xgrad_arr) {
        mat.create(rows, cols, type);
        data_size = rows * cols * mat.elemSize();
        ifs.read((char *)mat.ptr(), data_size);
    }

    ifs.read((char *)&size, sizeof(int));
    this->vert_epi_ygrad_arr.resize(size);

    ifs.read((char *)&cols, sizeof(int));
    ifs.read((char *)&rows, sizeof(int));
    ifs.read((char *)&type, sizeof(int));

    for(cv::Mat& mat: this->vert_epi_ygrad_arr) {
        mat.create(rows, cols, type);
        data_size = rows * cols * mat.elemSize();
        ifs.read((char *)mat.ptr(), data_size);
    }

    ifs.close();
}
*/
