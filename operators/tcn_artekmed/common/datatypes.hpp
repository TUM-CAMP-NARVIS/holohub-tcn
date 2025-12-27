#pragma once

#include <yaml-cpp/yaml.h>
#include <Eigen/Core>
#include <Eigen/src/Geometry/Quaternion.h>
#include <Eigen/src/Geometry/AngleAxis.h>
#include <Eigen/src/Geometry/RotationBase.h>
#include <Eigen/LU>

#include <cuda_runtime.h>

#include <gxf/multimedia/camera.hpp>

// Simple math structs compatible with CUDA device code
// struct float2 { float x, y; };
// struct float3 { float x, y, z; };
// struct float4 { float x, y, z, w; };

struct float4x4 {
  // Column-major like in original code
  float4 col[4];
};

// Vertex structure (kept for parity, used internally)
struct VertexPositionTexcoord {
  float3 position;
  float2 texcoord;
};

// Camera model parameters (Brown–Conrady as used in the original)
struct CameraParameters {
  float fx{0}, fy{0}, cx{0}, cy{0};
  float k1{0}, k2{0}, k3{0}, k4{0}, k5{0}, k6{0};
  float codx{0}, cody{0};
  float p1{0}, p2{0};
  int is_distorted{0};  // 1 = distorted, 0 = undistorted
};

struct RigidTransform {
  Eigen::Vector3f translation;
  Eigen::Quaternion<float> rotation;

  RigidTransform() = default;
  RigidTransform(Eigen::Vector3f _translation, Eigen::Quaternion<float> _rotation)
      : translation(std::move(_translation)), rotation(std::move(_rotation)) {}

  RigidTransform(const RigidTransform& other) = default;

  bool operator==(const RigidTransform& other) const {
    return translation == other.translation && rotation.w() == other.rotation.w() &&
           rotation.vec() == rotation.vec();
  }

  RigidTransform inverse() const {
    RigidTransform p(-translation, rotation.inverse());
    return p;
  }

  void toMatrix4f(Eigen::Matrix4f& result) const {
    auto R = rotation.normalized().toRotationMatrix();
    Eigen::Vector3f T = translation;
    Eigen::Matrix4f Trans;  // Your Transformation Matrix
    Trans.setIdentity();    // Set to Identity to make bottom row of Matrix 0,0,0,1
    Trans.block<3, 3>(0, 0) = R;
    Trans.block<3, 1>(0, 3) = T;
    result = Trans;
  }

#ifndef __CUDACC__
  template <typename OStream>
  friend OStream& operator<<(OStream& os, const RigidTransform& c) {
    return os << "[RigidTransform translation=" << c.translation << std::endl
              << ", rotation=" << c.rotation << "]";
  }
#endif
};

// Helper to cast Eigen::Matrix4f (row-major) to our float4x4 (column array)
inline float4x4 float4x4Cast(const Eigen::Matrix4f& m) {
  float4x4 out{};
  // Fill columns
  for (int c = 0; c < 4; ++c) {
    out.col[c].x = m(0, c);
    out.col[c].y = m(1, c);
    out.col[c].z = m(2, c);
    out.col[c].w = m(3, c);
  }
  return out;
}


//YAML Converters
namespace YAML {

// Eigen::Vector3f
template <>
struct convert<Eigen::Vector3f> {
  static Node encode(const Eigen::Vector3f& rhs) {
    Node node;
    node.push_back(rhs.x());
    node.push_back(rhs.y());
    node.push_back(rhs.z());
    return node;
  }

  static bool decode(const Node& node, Eigen::Vector3f& rhs) {
    if (!node.IsSequence() || node.size() != 3) return false;
    rhs << node[0].as<float>(), node[1].as<float>(), node[2].as<float>();
    return true;
  }
};

// Eigen::Quaternionf
template <>
struct convert<Eigen::Quaternionf> {
  static Node encode(const Eigen::Quaternionf& rhs) {
    Node node;
    node.push_back(rhs.w());  // scalar first (w, x, y, z)
    node.push_back(rhs.x());
    node.push_back(rhs.y());
    node.push_back(rhs.z());
    return node;
  }

  static bool decode(const Node& node, Eigen::Quaternionf& rhs) {
    if (!node.IsSequence() || node.size() != 4) return false;
    rhs.coeffs() << node[0].as<float>(), node[1].as<float>(),
                     node[2].as<float>(), node[3].as<float>();
    return true;
  }
};

// RigidTransform
template <>
struct convert<RigidTransform> {
  static Node encode(const RigidTransform& rhs) {
    Node node;
    node["translation"] = Node(convert<Eigen::Vector3f>::encode(rhs.translation));
    node["rotation"] = Node(convert<Eigen::Quaternionf>::encode(rhs.rotation));
    return node;
  }

  static bool decode(const Node& node, RigidTransform& rhs) {
    if (!node.IsMap() || node.size() != 2) return false;

    const auto& trans_node = node["translation"];
    const auto& rot_node = node["rotation"];

    if (!trans_node.IsSequence() || trans_node.size() != 3 ||
        !rot_node.IsSequence() || rot_node.size() != 4) {
      return false;
        }

    rhs.translation << trans_node[0].as<float>(), trans_node[1].as<float>(), trans_node[2].as<float>();
    rhs.rotation.coeffs() << rot_node[0].as<float>(), rot_node[1].as<float>(),
                             rot_node[2].as<float>(), rot_node[3].as<float>();
    return true;
  }
};

// CameraParameters

template <>
struct convert<CameraParameters> {
  static Node encode(const CameraParameters& rhs) {
    Node node;
    node["fx"] = rhs.fx;
    node["fy"] = rhs.fy;
    node["cx"] = rhs.cx;
    node["cy"] = rhs.cy;
    node["k1"] = rhs.k1; node["k2"] = rhs.k2; node["k3"] = rhs.k3;
    node["k4"] = rhs.k4; node["k5"] = rhs.k5; node["k6"] = rhs.k6;
    node["codx"] = rhs.codx; node["cody"] = rhs.cody;
    node["p1"] = rhs.p1; node["p2"] = rhs.p2;
    node["is_distorted"] = rhs.is_distorted;
    return node;
  }

  static bool decode(const Node& node, CameraParameters& rhs) {
    if (!node.IsMap()) return false;

    rhs.fx = node["fx"].as<float>(rhs.fx);
    rhs.fy = node["fy"].as<float>(rhs.fy);
    rhs.cx = node["cx"].as<float>(rhs.cx);
    rhs.cy = node["cy"].as<float>(rhs.cy);
    rhs.k1 = node["k1"].as<float>(rhs.k1);
    rhs.k2 = node["k2"].as<float>(rhs.k2);
    rhs.k3 = node["k3"].as<float>(rhs.k3);
    rhs.k4 = node["k4"].as<float>(rhs.k4);
    rhs.k5 = node["k5"].as<float>(rhs.k5);
    rhs.k6 = node["k6"].as<float>(rhs.k6);
    rhs.codx = node["codx"].as<float>(rhs.codx);
    rhs.cody = node["cody"].as<float>(rhs.cody);
    rhs.p1 = node["p1"].as<float>(rhs.p1);
    rhs.p2 = node["p2"].as<float>(rhs.p2);
    rhs.is_distorted = node["is_distorted"].as<int>(rhs.is_distorted);
    return true;
  }
};

// gxf Vector2
template <>
struct convert<nvidia::gxf::Vector2f> {
  static Node encode(const nvidia::gxf::Vector2f& rhs) {
    Node node;
    node.push_back(rhs.x);
    node.push_back(rhs.y);
    return node;
  }

  static bool decode(const Node& node, nvidia::gxf::Vector2f& rhs) {
    if (!node.IsSequence() || node.size() != 2) return false;
    rhs.x = node[0].as<float>();
    rhs.y = node[1].as<float>();
    return true;
  }
};

template <>
struct convert<nvidia::gxf::Vector2u> {
  static Node encode(const nvidia::gxf::Vector2u& rhs) {
    Node node;
    node.push_back(rhs.x);
    node.push_back(rhs.y);
    return node;
  }

  static bool decode(const Node& node, nvidia::gxf::Vector2u& rhs) {
    if (!node.IsSequence() || node.size() != 2) return false;
    rhs.x = node[0].as<uint32_t>();
    rhs.y = node[1].as<uint32_t>();
    return true;
  }
};


// gxf DistortionType (should be improved..)
template <>
struct convert<nvidia::gxf::DistortionType> {
  static Node encode(const nvidia::gxf::DistortionType& rhs) {
    return Node(static_cast<int>(rhs));
  }

  static bool decode(const Node& node, nvidia::gxf::DistortionType& rhs) {
    if (!node.IsScalar()) return false;
    rhs = static_cast<nvidia::gxf::DistortionType>(node.as<int>());
    return true;
  }
};

// gxf CameraModel
template <>
struct convert<nvidia::gxf::CameraModel> {
  static Node encode(const nvidia::gxf::CameraModel& rhs) {
    Node node;
    node["dimensions"] = rhs.dimensions;           // nvidia::gxf::Vector2u
    node["focal_length"] = rhs.focal_length;       // nvidia::gxf::Vector2f
    node["principal_point"] = rhs.principal_point; // nvidia::gxf::Vector2f
    node["skew_value"] = rhs.skew_value;
    node["distortion_type"] = rhs.distortion_type;

    Node dist_node;
    for (int i = 0; i < nvidia::gxf::CameraModel::kMaxDistortionCoefficients; ++i) {
      dist_node.push_back(rhs.distortion_coefficients[i]);
    }
    node["distortion_coefficients"] = dist_node;
    return node;
  }

  static bool decode(const Node& node, nvidia::gxf::CameraModel& rhs) {
    if (!node.IsMap()) return false;

    rhs.dimensions = node["dimensions"].as<nvidia::gxf::Vector2u>();
    rhs.focal_length = node["focal_length"].as<nvidia::gxf::Vector2f>();
    rhs.principal_point = node["principal_point"].as<nvidia::gxf::Vector2f>();
    rhs.skew_value = node["skew_value"].as<float>();
    rhs.distortion_type = node["distortion_type"].as<nvidia::gxf::DistortionType>();

    const auto& dist_node = node["distortion_coefficients"];
    if (dist_node.IsSequence() && dist_node.size() == nvidia::gxf::CameraModel::kMaxDistortionCoefficients) {
      for (int i = 0; i < nvidia::gxf::CameraModel::kMaxDistortionCoefficients; ++i) {
        rhs.distortion_coefficients[i] = dist_node[i].as<float>();
      }
    }
    return true;
  }
};

}  // namespace YAML