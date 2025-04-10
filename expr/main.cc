#include <fmt/format.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <CLI/CLI.hpp>
#include <fkYAML/node.hpp>
#include <fstream>
#include <thread>

#include "BoW3D.h"
#include "LinK3D_Extractor.h"
#include "expr-utils/data_loader.hh"

namespace fs = std::filesystem;

namespace {
std::string config_path{""};

struct Config {
  std::string dataset_type{""};
  std::string lidar_path{""};
  std::string pose_path{""};
  std::string calib_path{""};
  std::string output_path{""};

  // Parameters of LinK3D
  int nScans = 64;  // Number of LiDAR scan lines
  float scanPeriod = 0.1;
  float minimumRange = 0.1;
  float distanceTh = 0.4;
  int matchTh = 6;

  // Parameters of BoW3D
  float thr = 3.5;
  int thf = 5;
  int num_add_retrieve_features = 5;
};

int parse_param(int argc, char** argv) {
  CLI::App app{"STDesc expriement"};
  app.add_option("-c,--config", config_path, "Path to the config file")
      ->required();
  CLI11_PARSE(app, argc, argv);
  return 0;
}

Config load_config(const std::string& config_path) {
  Config ret;
  fkyaml::node node;
  {
    std::ifstream file(config_path);
    node = fkyaml::node::deserialize(file);
  }

  // Parameters of LinK3D
  ret.nScans = node["nScans"].get_value<int>();
  ret.scanPeriod = node["scanPeriod"].get_value<float>();
  ret.minimumRange = node["minimumRange"].get_value<float>();
  ret.distanceTh = node["distanceTh"].get_value<float>();
  ret.matchTh = node["matchTh"].get_value<int>();

  // Parameters of BoW3D
  ret.thr = node["thr"].get_value<float>();
  ret.thf = node["thf"].get_value<int>();
  ret.num_add_retrieve_features =
      node["num_add_retrieve_features"].get_value<int>();

  // Parameters of dataset
  ret.dataset_type = node["dataset"].get_value<std::string>();
  ret.lidar_path = node["lidar"].get_value<std::string>();
  ret.pose_path = node["pose"].get_value<std::string>();
  ret.calib_path = node["calib"].get_value<std::string>();
  ret.output_path = node["result"].get_value<std::string>();
  return ret;
}

struct Result {
  size_t key_frame_id;
  size_t loop_frame_id;
  double score;
  double iou;
  Eigen::Vector3f center;
  double t_desc;
  double t_query;

  friend std::ostream& operator<<(std::ostream& os, const Result& res) {
    os << res.key_frame_id << "," << res.loop_frame_id << "," << res.score
       << "," << res.iou << "," << res.center.x() << "," << res.center.y()
       << "," << res.center.z() << "," << res.t_desc << "," << res.t_query;
    return os;
  }
};

double iou(pcl::PointCloud<pcl::PointXYZI>::ConstPtr cld1,
           pcl::PointCloud<pcl::PointXYZI>::ConstPtr cld2) {
  pcl::KdTreeFLANN<pcl::PointXYZI> tree;
  tree.setInputCloud(cld1);
  std::vector<bool> marks1(cld1->size(), false);
  std::vector<bool> marks2(cld2->size(), false);

  // search the near points in 0.5m
  for (size_t i = 0; i < cld2->size(); i++) {
    const auto& p = cld2->at(i);
    std::vector<int> indices;
    std::vector<float> distances;
    tree.radiusSearch(p, 0.5, indices, distances);
    if (indices.empty()) {
      continue;
    }
    // mark the indices
    for (const auto& idx : indices) {
      marks1[idx] = true;
    }
    marks2[i] = true;
  }

  // calculate the iou, no need to minus the intersection, because this is point
  // num, not volume
  size_t inter = std::count(marks1.begin(), marks1.end(), true) +
                 std::count(marks2.begin(), marks2.end(), true);
  return double(inter) / double(cld1->size() + cld2->size());
}

struct TimeCounter {
  std::chrono::high_resolution_clock::time_point start;

  TimeCounter() : start(std::chrono::high_resolution_clock::now()) {}
  double duration() {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
               .count() /
           1000.0;
  }
};
}  // namespace

int main(int argc, char** argv) {
  // parse parameters
  parse_param(argc, argv);
  auto cfg = load_config(config_path);

  // data loader
  auto loader = utils::create_loader(cfg.dataset_type, cfg.lidar_path,
                                     cfg.pose_path, cfg.calib_path);

  // Link3D
  auto pLinK3dExtractor =
      new BoW3D::LinK3D_Extractor(cfg.nScans, cfg.scanPeriod, cfg.minimumRange,
                                  cfg.distanceTh, cfg.matchTh);

  // BoW3D
  auto pBoW3D = new BoW3D::BoW3D(pLinK3dExtractor, cfg.thr, cfg.thf,
                                 cfg.num_add_retrieve_features);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cur_cld;
  std::deque<Result> results;
  std::vector<utils::CloudT::Ptr> clouds;
  clouds.reserve(loader->size());
  fmt::print(
      "#key_id,point_num,loop_id,score,iou,desc-ms,query-ms,update-ms\n");
  for (size_t i = 0; i < loader->size(); i++) {
    auto tmp = loader->seq(i, true);
    clouds.push_back(tmp);
    Eigen::Vector4f center;
    pcl::compute3DCentroid(*tmp, center);
    cur_cld.reset(new pcl::PointCloud<pcl::PointXYZ>);
    cur_cld->resize(tmp->size());
    std::transform(tmp->begin(), tmp->end(), cur_cld->begin(),
                   [](const auto& p) { return pcl::PointXYZ(p.x, p.y, p.z); });
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() = center.head<3>();
    pcl::transformPointCloud(*cur_cld, *cur_cld, transform.inverse());
    fmt::print("{},{},", i, cur_cld->size());

    TimeCounter tc_desc;
    auto pCurrentFrame = new BoW3D::Frame(pLinK3dExtractor, cur_cld);
    auto t_desc_ms = tc_desc.duration();
    if (i > 2) {
      int loopFrameId = -1;
      Eigen::Matrix3d loopRelR;
      Eigen::Vector3d loopRelt;
      TimeCounter tc_query;
      auto bow3d_res =
          pBoW3D->retrieve(pCurrentFrame, loopFrameId, loopRelR, loopRelt);
      auto t_query_ms = tc_query.duration();
      if (bow3d_res.loop_frame_id != -1) {
        auto cld1 = clouds[i];
        auto cld2 = clouds[bow3d_res.loop_frame_id];
        Result res;
        res.key_frame_id = i;
        res.loop_frame_id = bow3d_res.loop_frame_id;
        res.score = 1.0 / (bow3d_res.loop_rel_t.norm() + 1.0);
        res.iou = iou(cld1, cld2);
        res.t_desc = t_desc_ms;
        res.t_query = t_query_ms;
        utils::CloudT ctr_cld;
        ctr_cld += *cld1;
        ctr_cld += *cld2;
        pcl::compute3DCentroid(ctr_cld, center);
        res.center = center.head<3>();
        results.push_back(res);
        fmt::print("{},{},{},", i, res.score, res.iou);
      } else {
        fmt::print("-1,-1,-1,");
      }
    } else {
      fmt::print("-1,-1,-1,");
    }

    if (pCurrentFrame) {
      TimeCounter tc_update;
      pBoW3D->update(pCurrentFrame);
      auto t_update_ms = tc_update.duration();
      fmt::print("{}\n", t_update_ms);
    } else {
      fmt::print("-1\n");
    }
    std::cout << std::flush;
  }

  // save the results
  std::ofstream ofs(cfg.output_path);
  for (const auto& res : results) {
    ofs << res << std::endl;
  }

  return 0;
}