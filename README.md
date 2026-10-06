# VolkMath
A header-only vector, matrix, and angle math library for C++20.

### Currently supports:
- **Vectors**
  - `Vec2`, `Vec3`, and `Vec4` templated on any floating-point type, with `f` and `d` aliases
  - Arithmetic operators, indexing, length, distance, and zero/NaN checks
  - Dot and cross products, and normalization

- **Quaternions & matrices**
  - Quaternion multiplication, conjugate, and vector rotation on `Vec4`
  - `Mat3x3` built from a quaternion, with transpose, multiplication, and vector transforms
  - `Mat4x4` point transforms
  - `OrientedScale` and `Transform` (rotation + position)

- **Geometry** (`volk::math::geometry`)
  - Triangles with centers
  - AABBs with expansion, ray intersection, and longest axis

- **Angles** (`volk::math::angle`)
  - Degree/radian conversion and wrapping to ±180°
  - Quaternion to Euler angles
  - Pitch and yaw from one point to another, with clamping and normalization

## Usage

VolkMath is header-only. Add it as a submodule and put `external\VolkMath\include` in your project's **Additional Include Directories**:

```cpp
#include <VolkMath/math.hh>

using volk::math::Vec3f;

int main() {
    const Vec3f camera{ 0.0f, 1.8f, 0.0f };
    const Vec3f target{ 10.0f, 1.8f, 10.0f };

    auto angles = volk::math::angle::calculate_angles(camera, target);
    volk::math::angle::normalize_angles(angles);

    const float distance = camera.distance(target);
}
```

## Contributors
- **Creator:** [lyk64](https://github.com/lyk64)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
