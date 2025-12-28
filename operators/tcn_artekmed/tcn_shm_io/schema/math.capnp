@0xbde9e6bfd684cb8e;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("artekmed::schema");

struct Vec3b {
	x @0 :UInt8;
	y @1 :UInt8;
	z @2 :UInt8;
}

struct Vec4b {
	x @0 :UInt8;
	y @1 :UInt8;
	z @2 :UInt8;
	w @3 :UInt8;
}

struct Vec2f {
	x @0 :Float32;
	y @1 :Float32;
}

struct Vec3f {
	x @0 :Float32;
	y @1 :Float32;
	z @2 :Float32;
}

struct Vec4f {
	x @0 :Float32;
	y @1 :Float32;
	z @2 :Float32;
	w @3 :Float32;
}

struct Vec2ui {
    v1 @0 :UInt32;
    v2 @1 :UInt32;
}

struct Vec3ui {
    v1 @0 :UInt32;
    v2 @1 :UInt32;
    v3 @2 :UInt32;
}

struct Vec4ui {
    v1 @0 :UInt32;
    v2 @1 :UInt32;
    v3 @2 :UInt32;
    v4 @3 :UInt32;
}


struct Matrix3x3f {
	col0 @0 :Vec3f;
	col1 @1 :Vec3f;
	col2 @2 :Vec3f;
}

struct Matrix3x4f {
	col0 @0 :Vec3f;
	col1 @1 :Vec3f;
	col2 @2 :Vec3f;
	col3 @3 :Vec3f;
}

struct Matrix4x4f {
	col0 @0 :Vec4f;
	col1 @1 :Vec4f;
	col2 @2 :Vec4f;
	col3 @3 :Vec4f;
}

struct GenericVector(Value) {
    size @0 :Int32;
    data @1 :List(Entry);
    struct Entry {
        value @0 :Value;
    }
}

struct GenericMatrix(Value) {
    rows @0 :Int32;
    cols @1 :Int32;
    data @2 :List(Entry);
    struct Entry {
        value @0 :Value;
    }
}

struct Quaternion {
	x @0 :Float32;
	y @1 :Float32;
	z @2 :Float32;
	w @3 :Float32;
}

# rename to RigidTransform ?
struct Pose {
	translation @0 :Vec3f;
	rotation    @1 :Quaternion;
}

struct Range3D {
    min @0 :Vec3f;
    max @1 :Vec3f;
}