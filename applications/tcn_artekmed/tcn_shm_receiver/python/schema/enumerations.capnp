@0xfec0d5c7fb840fd1;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("artekmed::schema");

annotation optional @0x8ae3f12dce834f5a (*) :Bool;
annotation namedItem @0x977c06dfa7c0a28b (*) :Bool;
annotation storeRaw @0xfbb53e22345e9a77 (*) :Bool;
annotation unionStruct @0xb1877c98977e8a71 (*) :Text;
annotation alternateKey @0x841cb8597af11ece (*) :Text;


enum BufferSpace {
    spaceAuto @0;
    spaceSystem @1;
    spaceCuda @2;
    spaceCudaHost @3;
    spaceCudaManaged @4;
    spaceShm @5;
}

enum TimeSynchronizationMode {
    earliest @0;
    latest @1;
    exact @2;
    window @3;
    windowEarliest @4;
    windowLatest @5;
}

enum CudaSyncStrategy {
    syncStrategyPerformance @0 $alternateKey("performance");
    syncStrategyNormal @1 $alternateKey("normal");
    syncStrategyPedantic @2 $alternateKey("pedantic");
}

enum DataType {
    scalar     @0;
    vec2f      @1;
    vec3f      @2;
    vec4f      @3;
    matrix3x3f @4;
    matrix3x4f @5;
    matrix4x4f @6;
    quaternion @7;
    pose       @8;
    humanPoseList  @9;
    image  @10;
    pointCloud @11;
    mesh @12;
    buffer @13;
}

enum BitstreamEncoding {
    h264 @0;
    h265 @1;
}

enum BitstreamStreamFormat {
    byteStream @0;
    avc @1;
}

enum BitstreamCompressionMethod {
    lossless @0;
    lossy @1;
    lowLatency @2;
    custom @3;
}

enum BitstreamPixelFormat {
    rgba32 @0;
    uint8 @1;
    uint16 @2;
    uint32 @3;
}

enum BitstreamCompressionProfile {
    undefined @0;
    h264Autoselect @1;
    h264Baseline @2;
    h264Main @3;
    h264High @4;
    h264High444 @5;
    h264Stereo @6;
    h264ProgressiveHigh @7;
    h264ConstrainedHigh @8;
    hevcMain @9;
    hevcMain10 @10;
    hevcFrext @11;
}

enum BitstreamCompressionPreset {
    undefined @0;
    p1 @1;
    p2 @2;
    p3 @3;
    p4 @4;
    p5 @5;
    p6 @6;
    p7 @7;
}

enum BitstreamCompressionTuningInfo {
    undefined @0;
    highQuality @1;
    lowLatency @2;
    ultraLowLatency @3;
    lossless @4;
}

enum BitstreamCompressionRcMode {
    undefined @0;
    rcConstqp @1;
    rcVbr @2;
    rcCbr @3;
}

enum PixelFormat {
    unknown @0;
    luminance @1;
    rgb @2;
    bgr @3;
    rgba @4;
    bgra @5;
    yuv422 @6;
    yuv411 @7;
    raw @8;
    depth @9;
    float @10;
    mjpeg @11;
    zdepth @12;
    nv12 @13;
    h264 @14;
    h265 @15;
}


enum PointCloudFormat {
    positionNormalRadius @0;
    positionTextureCoordinate @1;
}

enum MtcFrameRate {
    frameRate24 @0;
    frameRate25 @1;
    frameRate29 @2;
    frameRate30 @3;
}

enum SensorType {
    unknown @0;
    azureKinectCamera @1;
    orbbecCamera @2;
}

enum SensorConnectionType {
    usbc @0;
    network @1;
}

enum CameraPortType {
    unknown @0;
    pointxyzuv @1;
    depthimage @2;
    depthweights @3;
    colorimage @4;
    infraredimage @5;
    maskimage @6;
    bodytrackingPoses @7;
    bodytrackingIndex @8;
    surfel @9;
    textureCoordinates @10;
    tcCameraIds @11;
    meshVertex @12;
    meshFace @13;
    dracoMesh @14;
    h264Bitstream @15;
    h265Bitstream @16;
    surfelOctree @17;
    imuSample @18;
    meshBitstream @19;
}

enum StreamProducerType {
    nullReceiver @0;
    fileReader @1;
    deviceReader @2;
    zmqSubscriber @3;
    shmSubscriber @4;
    ddsSubscriber @5;
    udpReceiver @6;
    rudpReceiver @7;
    rtspReceiver @8;
    rtpReceiver @9;
    live555RtspReceiver @10;
    avbtReceiver @11;
    gstreamerReceiver @12;
}

enum StreamConsumerType {
    nullSender @0;
    fileWriter @1;
    zmqPublisher @2;
    shmPublisher @3;
    ddsPublisher @4;
    udpSender @5;
    rudpSender @6;
    rtspSender @7;
    rtpSender @8;
    live555RtspSender @9;
    avbtSender @10;
    gstreamerSender @11;
}

enum StreamDataType {
    unknown @0;
    image @1;
    bitstream @2;
    positionVertex @3;
    positionVertexNormal @4;
    meshVertex @5;
    meshFace @6;
    cameraIds @7;
    textureCoordinates @8;
    dracoMesh @9;
    bodytrackingPoses @10;
    imuSample @11;
    meshBitstream @12;
}

enum ShmConnectionStatus {
    open @0;
    closed @1;
}



enum ScalarType {
    char          @0;
    unsignedChar  @1;
    wchar         @2;
    unsignedWchar @3;
    int4          @4;
    int8          @5;
    int16         @6;
    int32         @7;
    int64         @8;
    uint4         @9;
    uint8         @10;
    uint16        @11;
    uint32        @12;
    uint64        @13;
    float16       @14;
    float32       @15;
    float64       @16;
    bool          @17;
}

enum MemoryRepresentationType {
    raw        @0;
    compressed @1;
}

enum ContainerType {
    none    @0;
    array1d @1;
    array2d @2;
    array3d @3;
}

enum CardinalityType {
    fixed    @0;
    variable @1;
}

# Image Header Type

enum ImageFormatType {
    luminance @0;
    depth     @1;
    rgb       @2;
    bgr       @3;
    rgba      @4;
    bgra      @5;
    hsv       @6;
    lab       @7;
    yuv422    @8;
    yuv411    @9;
    nv12      @10;
}

enum ImageCompressionType {
    none    @0;
    jpeg    @1;
    h264    @2;
    h265    @3;
    zstd    @4;
    rvl     @5;
    trvl    @6;
    zdepth  @7;
}

enum GeometryFormatType {
    point             @0;
    anchor            @1;
    sprite            @2;
    surfaceMesh       @3;
    spline            @4;
    nurbs             @5;
    voxelGrid         @6;
    sparseVoxelGrid   @7;
}

enum GeometryCompressionType {
    none @0;
    draco @1;
}


enum TransformFormatType {
    none                @0;
    translation         @1;
    rotation            @2;
    scaling             @3;
    rigidTransform      @4;
    similarityTransform @5;
    affineTransform     @6;
    projectiveTransform @7;
}

enum TransformListModel {
    independent         @0;
    azureKinectBody     @1;
}



enum ApplicationLifeCycleEventDomain {
  application @0;
  coreService @1;
  consumerService @2;
  producerService @3;
  runtime @4;
  dataflow @5;
  component @6;
}

enum ApplicationLifeCycleStatus {
  initialized @0;
  started @1;
  stopped @2;
  uninitialized @3;
}

enum ApplicationLifeCycleRequest {
  requestShutdown @0;
  playbackStart @1;
  playbackPause @2;
  playbackStep @3;
  playbackStop @4;
  recordingStart @5;
  recordingPause @6;
  recordingStop @7;
}

enum HealthStatusUpdate {
  watchdogEvent @0;
  captureDeviceTimeout @1;
  diskSpaceWarning @2;
  lowLightWarning @3;
  lowFramerateWarning @4;
  writeQueueWarning @5;
  framerateInfo @6;
  imuMovementWarning @7;
}

enum CameraDepthMode {
    cameraK4aDepthModeOff @0;
    cameraK4aDepthModeNfov2x2binned @1;
    cameraK4aDepthModeNfovUnbinned @2;
    cameraK4aDepthModeWfov2x2binned @3;
    cameraK4aDepthModeWfovUnbinned @4;
    cameraK4aDepthModePassiveIr @5;
}

enum CameraColorResolution {
    cameraK4aColorResolutionOff @0;
    cameraK4aColorResolution720p @1;
    cameraK4aColorResolution1080p @2;
    cameraK4aColorResolution1440p @3;
    cameraK4aColorResolution1536p @4;
    cameraK4aColorResolution2160p @5;
    cameraK4aColorResolution3072p @6;
}



enum DataflowTraceNodeType {
    component @0;
    service @1;
    producer @2;
    consumer @3;
}

# dataflow components default events
enum DataflowTraceComponentEventType {
    unknown @0;
    initializeBegin @1;
    connectionStatus @2;
    onInputsConnected @3;
    initializeEnd @4;
    start @5;
    onInputsStartSending @6;
    doSendBegin @7;
    doSendEnd @8;
    doProcessBegin @9;
    doProcessEnd @10;
    parallelDoProcessBegin @11;
    parallelDoProcessEnd @12;
    handleReceiveBegin  @13;
    handleReceiveEnd @14;
    stop @15;
    onInputsStopSending @16;
    teardown @17;
    onInputsDisconnected @18;
    removePort @19;
    destructor @20;
}

enum BodyJointType {
    pelvis @0;
    spineNaval @1;
    spineChest @2;
    neck @3;
    clavicleLeft @4;
    shoulderLeft @5;
    elbowLeft @6;
    wristLeft @7;
    handLeft @8;
    handtipLeft @9;
    thumbLeft @10;
    clavicleRight @11;
    shoulderRight @12;
    elbowRight @13;
    wristRight @14;
    handRight @15;
    handtipRight @16;
    thumbRight @17;
    hipLeft @18;
    kneeLeft @19;
    ankleLeft @20;
    footLeft @21;
    hipRight @22;
    kneeRight @23;
    ankleRight @24;
    footRight @25;
    head @26;
    nose @27;
    eyeLeft @28;
    earLeft @29;
    eyeRight @30;
    earRight @31;
    count @32;
}

