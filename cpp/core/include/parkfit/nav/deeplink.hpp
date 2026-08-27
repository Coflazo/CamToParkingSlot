// SPDX-License-Identifier: MIT
//
// Handing a parking space to whatever navigation app the driver already uses.
//
// The whole point of this file is one rule: **hand over coordinates, never an address.**
//
// It is tempting to pass "Prinsengracht 263, Amsterdam" and let Google sort it out. That
// throws away everything the product knows. Amsterdam publishes the bay as a polygon
// surveyed to the centimetre; a street address resolves to a building, and the building
// is on one side of a canal while the bay is on the other. Worse, the receiving app
// re-geocodes the text with its own database, so the same string can land in two
// different places in two different apps, and neither is the space we measured.
//
// So every link carries a latitude and longitude, printed with enough decimal places that
// the number stops being the limiting factor.
//
// **What is and is not exact.** The destination is exact: it is the surveyed point, passed
// through without a round trip. Where the *driver* is standing is not, and cannot be made
// so from here. A phone's GPS fix in a city centre is good to a few metres, and worse
// between tall buildings, which is why the origin is optional on every provider below.
// Leaving it out lets the navigation app use its own best fix rather than a stale one
// forwarded from a web page.
//
// For a car park the destination is the **entrance**, not the centroid. Routing a driver
// to the middle of a building is the classic parking-app failure: the centroid of an
// Amsterdam garage is frequently a canal, a tram line, or the wrong end of a one-way
// street. The caller is responsible for passing an entrance when one is known; this file
// only makes it impossible to forget by naming the field.

#pragma once

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace parkfit::nav {

/// Decimal places used when printing a coordinate.
///
/// Seven places is about 1.1 cm of latitude. That is far finer than any surveyed bay
/// corner and far finer than any consumer GPS, which is the point: the formatting must
/// never be the thing that loses precision, so the only error left is the one the physical
/// world imposes.
constexpr int kCoordinateDecimals = 7;

/// Navigation apps this project knows how to hand off to.
enum class NavProvider { GoogleMaps, AppleMaps, Waze, Yandex, OpenStreetMap, GeoUri };

inline const char* to_string(NavProvider provider) {
    switch (provider) {
        case NavProvider::GoogleMaps: return "google_maps";
        case NavProvider::AppleMaps: return "apple_maps";
        case NavProvider::Waze: return "waze";
        case NavProvider::Yandex: return "yandex";
        case NavProvider::OpenStreetMap: return "openstreetmap";
        case NavProvider::GeoUri: return "geo";
    }
    return "unknown";
}

inline const char* display_name(NavProvider provider) {
    switch (provider) {
        case NavProvider::GoogleMaps: return "Google Maps";
        case NavProvider::AppleMaps: return "Apple Maps";
        case NavProvider::Waze: return "Waze";
        case NavProvider::Yandex: return "Yandex Maps";
        case NavProvider::OpenStreetMap: return "OpenStreetMap";
        case NavProvider::GeoUri: return "Default map app";
    }
    return "unknown";
}

inline std::array<NavProvider, 6> all_providers() {
    return {NavProvider::GoogleMaps, NavProvider::AppleMaps,    NavProvider::Waze,
            NavProvider::Yandex,     NavProvider::OpenStreetMap, NavProvider::GeoUri};
}

/// Where to send the driver.
struct NavTarget {
    double lat{};
    double lon{};
    /// Shown as the pin label where the provider supports one. Never used for lookup.
    std::string label;
    /// True when `lat`/`lon` is a surveyed entrance rather than a centroid or a midpoint.
    /// Reported to the caller so an interface can say which it is instead of implying a
    /// precision the data does not have.
    bool is_entrance{false};

    [[nodiscard]] bool valid() const {
        return std::isfinite(lat) && std::isfinite(lon) && lat >= -90.0 && lat <= 90.0 &&
               lon >= -180.0 && lon <= 180.0;
    }
};

/// Optional starting point. Absent means "let the app use its own position fix".
struct NavOrigin {
    double lat{};
    double lon{};
    bool present{false};

    [[nodiscard]] bool valid() const {
        return present && std::isfinite(lat) && std::isfinite(lon) && lat >= -90.0 &&
               lat <= 90.0 && lon >= -180.0 && lon <= 180.0;
    }
};

/// Print a coordinate at full precision, with no exponent and no locale surprises.
///
/// snprintf with an explicit format rather than std::to_string, which gives six digits and
/// would silently round away about a metre, and rather than ostream, which honours the
/// global locale and would emit a comma decimal separator in a Dutch locale. A URL
/// containing "52,3730" is not a coordinate, it is two coordinates.
inline std::string format_coordinate(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.*f", kCoordinateDecimals, value);

    // Trim trailing zeros so a round number reads as one, keeping at least one decimal.
    std::string out(buffer);
    const std::size_t dot = out.find('.');
    if (dot != std::string::npos) {
        std::size_t last = out.find_last_not_of('0');
        if (last == dot) last = dot + 1;
        out.erase(last + 1);
    }
    return out;
}

inline std::string format_pair(double lat, double lon) {
    return format_coordinate(lat) + "," + format_coordinate(lon);
}

/// Percent-encode for a query-string value.
///
/// Only the unreserved set from RFC 3986 survives unescaped. Labels come from OpenStreetMap
/// and municipal data and contain spaces, ampersands and apostrophes, any of which would
/// otherwise end the parameter early and hand the app a truncated link.
inline std::string url_encode(const std::string& text) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    out.reserve(text.size() * 3 / 2);
    for (const unsigned char c : text) {
        const bool unreserved = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                                (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' ||
                                c == '~';
        if (unreserved) {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 0x0F];
        }
    }
    return out;
}

/// Build a driving-directions URL for one provider.
///
/// Returns an empty string when the target is not a usable coordinate, so a caller cannot
/// accidentally render a link that navigates to the Gulf of Guinea.
inline std::string build_url(NavProvider provider, const NavTarget& target,
                             const NavOrigin& origin = {}) {
    if (!target.valid()) return {};

    const std::string dest = format_pair(target.lat, target.lon);
    const bool with_origin = origin.valid();
    const std::string from = with_origin ? format_pair(origin.lat, origin.lon) : std::string{};

    switch (provider) {
        case NavProvider::GoogleMaps: {
            // The documented, version-pinned form. Without api=1 the older URL shapes are
            // interpreted differently across web, Android and iOS.
            std::string url = "https://www.google.com/maps/dir/?api=1&destination=" + dest +
                              "&travelmode=driving";
            if (with_origin) url += "&origin=" + from;
            return url;
        }
        case NavProvider::AppleMaps: {
            // dirflg=d selects driving. saddr is omitted rather than left empty, because an
            // empty saddr is what tells Apple Maps to use the current location.
            std::string url = "https://maps.apple.com/?daddr=" + dest + "&dirflg=d";
            if (with_origin) url += "&saddr=" + from;
            if (!target.label.empty()) url += "&q=" + url_encode(target.label);
            return url;
        }
        case NavProvider::Waze: {
            // Waze always navigates from the device's own position, so an origin is not
            // expressible and is deliberately dropped rather than silently mangled.
            return "https://waze.com/ul?ll=" + dest + "&navigate=yes";
        }
        case NavProvider::Yandex: {
            // rtext takes point~point; with one point it is treated as the destination.
            std::string url = "https://yandex.com/maps/?rtext=";
            if (with_origin) url += from;
            url += "~" + dest;
            url += "&rtt=auto";
            return url;
        }
        case NavProvider::OpenStreetMap: {
            std::string url = "https://www.openstreetmap.org/directions?engine=fossgis_osrm_car";
            url += "&route=";
            if (with_origin) url += from;
            url += ";" + dest;
            return url;
        }
        case NavProvider::GeoUri: {
            // RFC 5870. Whatever the device has registered as its map handler picks this up,
            // which is the right answer when the driver's preferred app is not on the list.
            std::string url = "geo:" + dest + "?q=" + dest;
            if (!target.label.empty()) url += "(" + url_encode(target.label) + ")";
            return url;
        }
    }
    return {};
}

/// One rendered handoff option.
struct NavLink {
    std::string provider;      ///< stable identifier, for logging and preferences
    std::string display_name;  ///< what to put on the button
    std::string url;
};

/// Every provider that can express this target, in the order an interface should show them.
inline std::vector<NavLink> build_links(const NavTarget& target, const NavOrigin& origin = {}) {
    std::vector<NavLink> links;
    if (!target.valid()) return links;

    for (const NavProvider provider : all_providers()) {
        std::string url = build_url(provider, target, origin);
        if (url.empty()) continue;
        links.push_back(NavLink{to_string(provider), display_name(provider), std::move(url)});
    }
    return links;
}

}  // namespace parkfit::nav
