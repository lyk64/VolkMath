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
        };

        template <std::floating_point T>
        struct Vec4 {
            T x{};
            T y{};
            T z{};
            T w{};
        };

        template <class T>
        struct Matrix4x4 {
            T m[4][4]{};
        };

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