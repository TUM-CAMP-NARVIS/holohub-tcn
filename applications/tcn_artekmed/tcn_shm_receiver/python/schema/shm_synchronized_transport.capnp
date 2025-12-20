@0xd7e5f1a2c3b4e5f6;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("artekmed::shm");

using import "enumerations.capnp".ShmConnectionStatus;
using import "enumerations.capnp".CameraPortType;
using import "enumerations.capnp".PixelFormat;
using import "enumerations.capnp".PointCloudFormat;
using import "enumerations.capnp".BitstreamEncoding;
using import "enumerations.capnp".BitstreamStreamFormat;
using import "enumerations.capnp".CameraDepthMode;
using import "enumerations.capnp".CameraColorResolution;
using import "enumerations.capnp".SensorType;

# --- From core.capnp ---

struct BufferInfo {
    frameSize @0 :UInt64 = 0;
    width      @1 :UInt32 = 0;
    height     @2 :UInt32 = 0;
    bitsPerElement @3 :UInt32 = 0;
    stride     @4 :UInt32 = 0;
    properties @5 :List(Entry);
    semanticType @6 :UInt64 = 0;

    struct Entry {
        name @0 :Text;
        value @1 :Int32;
        userData @2 :List(UserData);

        struct UserData {
            key @0: Text;
            value @1: Data;
        }
    }
}

# --- From measurement.capnp ---

struct StreamHeader {
    semanticType @20 :UInt64 = 0;
    dimX @0 :Int32 = 1;
    dimY @1 :Int32 = 1;
    dimZ @2 :Int32 = 1;
    bitsPerElement @3 :Int32;
    bufferLen @4: UInt64;
    timestamp @5: UInt64;
    deviceTimestamp @6: UInt64 = 0;
    deviceHostTimestamp @7: UInt64 = 0;

    union {
        scalar     @8 :Void;
        bitStream  :group {
            encoding @9: BitstreamEncoding;
            streamFormat @19: BitstreamStreamFormat;
        }
        image      @10 :PixelFormat;
        pointCloud @11 :PointCloudFormat;
        textureCoordinates @12 :Void;
        tcCameraIds @13 :Void;
        meshFaces   @14 :Void;
        meshVertex  @21 :Void;
        humanPoseList  @15 :Void;
        dracoMesh @16 :Void;
        buffer @17 :Void;
        imu @18 :Void;
    }
}

struct StreamFooter {
    union {
        null @0:Void;
        bitstream :group {
                byteCount @1 :UInt64;
            }
        positionVertex :group {
                boundingBox @2 :import "math.capnp".Range3D;
            }
        positionVertexNormalRadius :group {
                numOutVerts @3 :UInt64;
            }
        meshFace :group {
                numOutFaces @4 :UInt64;
            }
    }
}

# --- From deviceCalibration.capnp ---

struct DistortionParameters {
	k1 @0 :Float32;
	k2 @1 :Float32;
	k3 @2 :Float32;
	k4 @3 :Float32;
	k5 @4 :Float32;
	k6 @5 :Float32;
	tx @6 :Float32;
	ty @7 :Float32;
}

struct CameraIntrinsicParameters {
	fovX @0 :Float32;
	fovY @1 :Float32;
	cX   @2 :Float32;
	cY   @3 :Float32;
	width  @4 :Int32;
	height @5 :Int32;
	distortionParams @6 :DistortionParameters;
	intrinsicMatrix @7 :import "math.capnp".Matrix3x3f;
	metricRadius @8: Float32;
}

struct DeviceCalibration {
	depthCameraParameters @0 :CameraIntrinsicParameters;
	colorCameraParameters @1 :CameraIntrinsicParameters;
	color2depthTransform  @2 :import "math.capnp".Pose;
	cameraPose            @3 :import "math.capnp".Pose;
	isValid               @4 :Bool;
	depthMode             @5 :CameraDepthMode;
	colorResolution       @6 :CameraColorResolution;
	depth2accelTransform  @7 :import "math.capnp".Pose;
	depth2gyroTransform   @8 :import "math.capnp".Pose;
	rawCalibration        @9 :Data;
	serialNumber          @10 :Text;
	sensorType            @11 :SensorType = azureKinectCamera;
}

# --- From network.capnp ---

struct ShmConnectionStatusMessage {
    status @0: ShmConnectionStatus;
    bufferInfo @1 :BufferInfo;
    portType @2 :CameraPortType;
}

struct ShmStreamHeader {
    timestamp @0: UInt64;
    header  @1 :StreamHeader;
    footer @2 :StreamFooter;
}

struct ShmDeviceContext {
    name @0 :Text;
    depthUnitsPerMeter @1 :Float32;
    calibration @2 :DeviceCalibration;
    timestampOffset @3: UInt64;
    isValid @4: Bool;
    frameRate @5: Int32;
}

struct ShmBufferPortData {
    portType @0 :CameraPortType;
    metadata @1 :ShmStreamHeader;
    data @2: Data;
}

struct ShmBufferConnectionStatus {
    numPorts @0 :Int16;
    ports @1 :List(PortStatus);

    struct PortStatus {
        name @0 :Text;
        status @1 :ShmConnectionStatusMessage;
    }
}

struct ShmBufferDescriptor {
    timestamp @0: UInt64;
    numPorts @1 :Int16;
    ports @2 :List(PortEntry);

    struct PortEntry {
        name @0 :Text;
        data @1 :ShmBufferPortData;
    }
}
