#pragma once

#include <cmath>
#include <cassert>
#include <numbers>
#include <algorithm>

namespace Volk {
    namespace Math {
        template <std::floating_point T>
        struct Vec2 {
            T x{};
            T y{};

            constexpr Vec2() noexcept = default;
            constexpr Vec2(T x_, T y_) noexcept : x(x_), y(y_) {}

            [[nodiscard]] constexpr bool operator==(const Vec2&) const noexcept = default;

            [[nodiscard]] constexpr Vec2 operator-() const noexcept { return { -x, -y }; }

            [[nodiscard]] constexpr Vec2 operator+(const Vec2& other) const noexcept { return { x + other.x, y + other.y }; }
            constexpr Vec2& operator+=(const Vec2& other) noexcept { x += other.x; y += other.y; return *this; }

            [[nodiscard]] constexpr Vec2 operator-(const Vec2& other) const noexcept { return { x - other.x, y - other.y }; }
            constexpr Vec2& operator-=(const Vec2& other) noexcept { x -= other.x; y -= other.y; return *this; }

            [[nodiscard]] constexpr Vec2 operator*(T scalar) const noexcept { return { x * scalar, y * scalar }; }
            constexpr Vec2& operator*=(T scalar) noexcept { x *= scalar; y *= scalar; return *this; }

            [[nodiscard]] constexpr Vec2 operator/(T scalar) const noexcept { return { x / scalar, y / scalar }; }
            constexpr Vec2& operator/=(T scalar) noexcept { x /= scalar; y /= scalar; return *this; }

            [[nodiscard]] T& operator[](std::size_t i) { assert(i < 2); return *(&x + i); }
            [[nodiscard]] const T& operator[](std::size_t i) const { assert(i < 2); return *(&x + i); }

            [[nodiscard]] constexpr T length_squared() const noexcept { return x * x + y * y; }
            [[nodiscard]] T length() const noexcept { return std::sqrt(length_squared()); }
            [[nodiscard]] T distance(const Vec2& o) const noexcept { return (*this - o).length(); }

            [[nodiscard]] constexpr bool is_zero() const noexcept { return *this == Vec2{}; }

            [[nodiscard]] bool is_nan() const noexcept { return std::isnan(x) || std::isnan(y); }
        };

        template <std::floating_point T>
        struct Vec3 {
            T x{};
            T y{};
            T z{};

            constexpr Vec3() noexcept = default;
            constexpr Vec3(T x_, T y_, T z_) noexcept : x(x_), y(y_), z(z_) {}

            [[nodiscard]] constexpr bool operator==(const Vec3&) const noexcept = default;

            [[nodiscard]] constexpr Vec3 operator-() const noexcept { return { -x, -y, -z }; }

            [[nodiscard]] constexpr Vec3 operator+(const Vec3& other) const noexcept { return { x + other.x, y + other.y, z + other.z }; }
            constexpr Vec3& operator+=(const Vec3& other) noexcept { x += other.x; y += other.y; z += other.z; return *this; }

            [[nodiscard]] constexpr Vec3 operator-(const Vec3& other) const noexcept { return { x - other.x, y - other.y, z - other.z }; }
            constexpr Vec3& operator-=(const Vec3& other) noexcept { x -= other.x; y -= other.y; z -= other.z; return *this; }

            [[nodiscard]] constexpr Vec3 operator*(const Vec3& other) const noexcept { return { x * other.x, y * other.y, z * other.z }; }
            constexpr Vec3& operator*=(const Vec3& other) noexcept { x *= other.x; y *= other.y; z *= other.z; return *this; }

            [[nodiscard]] constexpr Vec3 operator*(T scalar) const noexcept { return { x * scalar, y * scalar, z * scalar }; }
            constexpr Vec3& operator*=(T scalar) noexcept { x *= scalar; y *= scalar; z *= scalar; return *this; }

            [[nodiscard]] constexpr Vec3 operator/(T scalar) const noexcept { return { x / scalar, y / scalar, z / scalar }; }
            constexpr Vec3& operator/=(T scalar) noexcept { x /= scalar; y /= scalar; z /= scalar; return *this; }

            [[nodiscard]] T& operator[](std::size_t i) { assert(i < 3); return *(&x + i); }
            [[nodiscard]] const T& operator[](std::size_t i) const { assert(i < 3); return *(&x + i); }

            [[nodiscard]] constexpr T length_squared() const noexcept { return x * x + y * y + z * z; }
            [[nodiscard]] T length() const noexcept { return std::sqrt(length_squared()); }
            [[nodiscard]] T distance(const Vec3& other) const noexcept { return (*this - other).length(); }

            [[nodiscard]] constexpr bool is_zero() const noexcept { return *this == Vec3{}; }

            [[nodiscard]] bool is_nan() const noexcept { return std::isnan(x) || std::isnan(y) || std::isnan(z); }

            static constexpr T dot(const Vec3& a, const Vec3& b) noexcept { return a.x * b.x + a.y * b.y + a.z * b.z; }
            constexpr T dot(const Vec3& o) const noexcept { return dot(*this, o); }

            static constexpr Vec3 cross(const Vec3& a, const Vec3& b) noexcept { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
            constexpr Vec3 cross(const Vec3& o) const noexcept { return cross(*this, o); }

            [[nodiscard]] Vec3 normalize() const noexcept {
                const T len = length();
                if (len > T(0)) return *this / len;
                return { T(0), T(0), T(0) };
            }
        };

        template <std::floating_point T>
        struct Vec4 {
            T x{};
            T y{};
            T z{};
            T w{};

            [[nodiscard]] constexpr Vec4 operator*(const Vec4& b) const noexcept {
                return {
                    w * b.x + x * b.w + y * b.z - z * b.y,
                    w * b.y - x * b.z + y * b.w + z * b.x,
                    w * b.z + x * b.y - y * b.x + z * b.w,
                    w * b.w - x * b.x - y * b.y - z * b.z
                };
            }

            [[nodiscard]] constexpr Vec4 conjugate() const noexcept {
                return { -x, -y, -z,  w };
            }

            [[nodiscard]] constexpr Vec3<T> rotate(const Vec3<T>& v) const noexcept {
                const Vec3<T> u{ x,y,z };
                const T s = w;
                const Vec3<T> t = u.cross(v) * T(2);
                return v + t * s + u.cross(t);
            }

            [[nodiscard]] constexpr Vec3<T> rotate_inv(const Vec3<T>& v) const noexcept {
                const Vec3<T> u{ x,y,z };
                const T s = w;
                const Vec3<T> t = u.cross(v) * T(2);
                return v - t * s + u.cross(t);
            }
        };

        template <std::floating_point T>
        struct Mat3x3 {
            Vec3<T> column0{};
            Vec3<T> column1{};
            Vec3<T> column2{};

            constexpr Mat3x3(const Vec3<T>& c0, const Vec3<T>& c1, const Vec3<T>& c2) noexcept : column0(c0), column1(c1), column2(c2) {}

            constexpr Mat3x3(const Vec4<T>& q) noexcept {
                const T x = q.x, y = q.y, z = q.z, w = q.w;
                const T x2 = x + x, y2 = y + y, z2 = z + z;

                const T xx = x2 * x, yy = y2 * y, zz = z2 * z;
                const T xy = x2 * y, xz = x2 * z, xw = x2 * w;
                const T yz = y2 * z, yw = y2 * w, zw = z2 * w;

                column0 = { T(1) - yy - zz, xy + zw, xz - yw };
                column1 = { xy - zw, T(1) - xx - zz, yz + xw };
                column2 = { xz + yw, yz - xw, T(1) - xx - yy };
            }

            [[nodiscard]] constexpr Mat3x3 get_transpose() const noexcept {
                return {
                    { column0.x, column1.x, column2.x },
                    { column0.y, column1.y, column2.y },
                    { column0.z, column1.z, column2.z }
                };
            }

            [[nodiscard]] constexpr Vec3<T> transform(const Vec3<T>& other) const noexcept {
                return column0 * other.x + column1 * other.y + column2 * other.z;
            }

            [[nodiscard]] constexpr Mat3x3 operator*(const Mat3x3& rhs) const noexcept {
                return { transform(rhs.column0), transform(rhs.column1), transform(rhs.column2) };
            }
        };

        template <std::floating_point T>
        struct Mat4x4 {
            Vec4<T> column0{};
            Vec4<T> column1{};
            Vec4<T> column2{};
            Vec4<T> column3{};
        };

        template <std::floating_point T>
        struct OrientedScale {
            Vec3<T> scale{ T(1), T(1), T(1) };
            Vec4<T> rotation{};

            [[nodiscard]] constexpr Mat3x3<T> to_mat3x3() const noexcept {
                const Mat3x3<T> rot{ rotation };
                Mat3x3<T> transpose = rot.get_transpose();
                transpose.column0 *= scale.x;
                transpose.column1 *= scale.y;
                transpose.column2 *= scale.z;
                return transpose * rot;
            }
        };

        template <std::floating_point T>
        struct Transform {
            Vec4<T> m_rotation{};
            Vec3<T> m_position{};

            [[nodiscard]] constexpr Vec3<T> transform_point(const Vec3<T>& point) const noexcept {
                return m_rotation.rotate(point) + m_position;
            }
        };

        namespace Geometry {
            template <std::floating_point T>
            struct Triangle {
                Vec3<T> v0{};
                Vec3<T> v1{};
                Vec3<T> v2{};

                [[nodiscard]] constexpr Vec3<T> center() const noexcept {
                    return (v0 + v1 + v2) / T{ 3 };
                }
            };

            template <std::floating_point T>
            struct AABB {
                Vec3<T> min{ std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity() };
                Vec3<T> max{ -std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity() };

                [[nodiscard]] constexpr Vec3<T> extent() const noexcept { return max - min; }

                constexpr void expand(const Vec3<T>& p) noexcept {
                    min.x = std::min(min.x, p.x); max.x = std::max(max.x, p.x);
                    min.y = std::min(min.y, p.y); max.y = std::max(max.y, p.y);
                    min.z = std::min(min.z, p.z); max.z = std::max(max.z, p.z);
                }

                [[nodiscard]] bool intersects(const Vec3<T>& origin, const Vec3<T>& dir, T& tmin, T& tmax) const noexcept {
                    tmin = T(0);
                    tmax = std::numeric_limits<T>::max();

                    for (int i = 0; i < 3; ++i) {
                        const T invD = T(1) / (&dir.x)[i];
                        T t0 = ((&min.x)[i] - (&origin.x)[i]) * invD;
                        T t1 = ((&max.x)[i] - (&origin.x)[i]) * invD;
                        if (invD < T(0)) std::swap(t0, t1);

                        tmin = t0 > tmin ? t0 : tmin;
                        tmax = t1 < tmax ? t1 : tmax;
                        if (tmax <= tmin) return false;
                    }
                    return true;
                }

                [[nodiscard]] constexpr int longest_axis() const noexcept {
                    const auto e = extent();
                    const std::array<T, 3> a{ e.x, e.y, e.z };
                    const auto it = std::max_element(a.begin(), a.end());
                    return static_cast<int>(it - a.begin());
                }
            };
        }

        namespace Angle {
            template <class T>
            [[nodiscard]] constexpr T rad_to_deg(T r) noexcept {
                return r * (T{ 180 } / std::numbers::pi_v<T>);
            }

            template <class T>
            [[nodiscard]] constexpr T deg_to_rad(T d) noexcept {
                return d * (std::numbers::pi_v<T> / T{ 180 });
            }

            template <std::floating_point T>
            [[nodiscard]] constexpr T wrap_deg180(T d) noexcept {
                return std::remainder(d, T{ 360 });
            }

            template <std::floating_point T>
            [[nodiscard]] Vec3<T> quat_to_euler_angles_deg(const Vec4<T>& q) noexcept {
                const T len2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
                if (len2 <= std::numeric_limits<T>::epsilon()) return { T{0}, T{0}, T{0} };

                const T invLen = T{ 1 } / std::sqrt(len2);
                const T x = q.x * invLen, y = q.y * invLen, z = q.z * invLen, w = q.w * invLen;

                const T sinp = T{ 2 } *(w * x - y * z);
                const T pitch = (std::fabs(sinp) >= T{ 1 })
                    ? std::copysign(std::numbers::pi_v<T> *T{ 0.5 }, sinp)
                    : std::asin(sinp);

                const T siny_cosp = T{ 2 } *(w * y + z * x);
                const T cosy_cosp = T{ 1 } - T{ 2 } *(x * x + y * y);
                const T yaw = std::atan2(siny_cosp, cosy_cosp);

                const T sinr_cosp = T{ 2 } *(w * z + x * y);
                const T cosr_cosp = T{ 1 } - T{ 2 } *(y * y + z * z);
                const T roll = std::atan2(sinr_cosp, cosr_cosp);

                return {
                    wrap_deg180(rad_to_deg(pitch)),
                    wrap_deg180(rad_to_deg(yaw)),
                    wrap_deg180(rad_to_deg(roll))
                };
            }

            template <std::floating_point T>
            inline Vec2<T> calculate_angles(const Vec3<T>& src, const Vec3<T>& dst) noexcept {
                Vec3<T> dir = src - dst;
                const T len = dir.length();
                if (len == T{}) return { T{}, T{} };

                const T pitch = std::asin(dir.y / len);
                const T yaw = -std::atan2(dir.x, -dir.z);

                return { rad_to_deg(pitch), rad_to_deg(yaw) };
            }

            template <std::floating_point T>
            constexpr Vec2<T>& normalize_angles(Vec2<T>& vector) noexcept {
                vector.x = std::clamp(vector.x, T{ -89 }, T{ 89 });

                while (vector.y > T{ 180 }) vector.y -= T{ 360 };
                while (vector.y <= T{ -180 }) vector.y += T{ 360 };

                return vector;
            }
        }
    }
}