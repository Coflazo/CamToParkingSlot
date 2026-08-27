// SPDX-License-Identifier: MIT
//
// Navigation handoff tests.
//
// These check the two things that actually break in production: coordinates losing
// precision on the way into a URL, and labels containing characters that end the query
// string early. Both fail silently and both send a driver somewhere real but wrong, which
// is the worst kind of bug this product can have.

#include "test_framework.hpp"

#include <string>

#include "parkfit/nav/deeplink.hpp"

using namespace parkfit::nav;

namespace {

/// A real Amsterdam bay corner, to seven decimals.
constexpr double kBayLat = 52.3675431;
constexpr double kBayLon = 4.8836219;

NavTarget bay() {
    NavTarget t;
    t.lat = kBayLat;
    t.lon = kBayLon;
    t.label = "Prinsengracht bay";
    return t;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

// ---------------------------------------------------------------------------
// Coordinate formatting
// ---------------------------------------------------------------------------
TEST_CASE("format: a coordinate keeps every digit that matters") {
    // std::to_string would give six decimals and round away roughly a metre.
    CHECK_EQ(format_coordinate(kBayLat), std::string("52.3675431"));
    CHECK_EQ(format_coordinate(kBayLon), std::string("4.8836219"));
}

TEST_CASE("format: seven decimals resolves about a centimetre") {
    // Two points 1.1 cm apart in latitude must not print identically, or the format is
    // the thing limiting accuracy rather than the survey.
    const std::string a = format_coordinate(52.3675431);
    const std::string b = format_coordinate(52.3675432);
    CHECK(a != b);
}

TEST_CASE("format: trailing zeros are trimmed but a decimal point survives") {
    CHECK_EQ(format_coordinate(52.0), std::string("52.0"));
    CHECK_EQ(format_coordinate(4.5000000), std::string("4.5"));
}

TEST_CASE("format: a negative coordinate keeps its sign") {
    CHECK_EQ(format_coordinate(-4.8836219), std::string("-4.8836219"));
}

TEST_CASE("format: never uses exponent notation") {
    // A tiny longitude near the prime meridian must not come out as 1e-07, which no
    // provider parses.
    const std::string small = format_coordinate(0.0000001);
    CHECK(!contains(small, "e"));
    CHECK(!contains(small, "E"));
}

TEST_CASE("format: a pair is comma separated with no space") {
    CHECK_EQ(format_pair(52.0, 4.5), std::string("52.0,4.5"));
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------
TEST_CASE("encode: characters that would truncate a query string are escaped") {
    // "Q-Park Bijenkorf & Dam" unescaped ends the parameter at the ampersand, and the
    // driver gets a link to a different place with no error anywhere.
    const std::string encoded = url_encode("Q-Park Bijenkorf & Dam");
    CHECK(!contains(encoded, " "));
    CHECK(!contains(encoded, "&"));
    CHECK(contains(encoded, "%20"));
    CHECK(contains(encoded, "%26"));
}

TEST_CASE("encode: unreserved characters are left alone") {
    CHECK_EQ(url_encode("Bay-17_A.b~c"), std::string("Bay-17_A.b~c"));
}

TEST_CASE("encode: non-ascii is percent encoded as utf-8 bytes") {
    // Dutch street names carry diaeresis; a raw byte in a URL is not portable.
    const std::string encoded = url_encode("IJdok\xc3\xab");
    CHECK(contains(encoded, "%C3%AB"));
}

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------
TEST_CASE("google: carries the exact destination and pins the api version") {
    const std::string url = build_url(NavProvider::GoogleMaps, bay());
    CHECK(contains(url, "api=1"));
    CHECK(contains(url, "destination=52.3675431,4.8836219"));
    CHECK(contains(url, "travelmode=driving"));
}

TEST_CASE("google: an origin is included when the driver's position is known") {
    NavOrigin origin;
    origin.lat = 52.3789;
    origin.lon = 4.9002;
    origin.present = true;

    const std::string url = build_url(NavProvider::GoogleMaps, bay(), origin);
    CHECK(contains(url, "origin=52.3789,4.9002"));
}

TEST_CASE("google: no origin parameter at all when the position is unknown") {
    // An empty origin= is not the same as an absent one: it can pin the route to nowhere
    // instead of letting the app use its own fix.
    const std::string url = build_url(NavProvider::GoogleMaps, bay());
    CHECK(!contains(url, "origin="));
}

TEST_CASE("apple: uses daddr and selects driving") {
    const std::string url = build_url(NavProvider::AppleMaps, bay());
    CHECK(contains(url, "daddr=52.3675431,4.8836219"));
    CHECK(contains(url, "dirflg=d"));
    CHECK(!contains(url, "saddr="));
}

TEST_CASE("waze: navigates from the device and ignores a supplied origin") {
    // Waze has no origin parameter. Silently dropping it is correct; pretending to honour
    // it would be worse.
    NavOrigin origin;
    origin.lat = 52.3789;
    origin.lon = 4.9002;
    origin.present = true;

    const std::string url = build_url(NavProvider::Waze, bay(), origin);
    CHECK(contains(url, "ll=52.3675431,4.8836219"));
    CHECK(contains(url, "navigate=yes"));
    CHECK(!contains(url, "52.3789"));
}

TEST_CASE("yandex: destination follows the tilde separator") {
    const std::string url = build_url(NavProvider::Yandex, bay());
    CHECK(contains(url, "rtext=~52.3675431,4.8836219"));
    CHECK(contains(url, "rtt=auto"));
}

TEST_CASE("openstreetmap: route runs origin semicolon destination") {
    NavOrigin origin;
    origin.lat = 52.3789;
    origin.lon = 4.9002;
    origin.present = true;

    const std::string url = build_url(NavProvider::OpenStreetMap, bay(), origin);
    CHECK(contains(url, "route=52.3789,4.9002;52.3675431,4.8836219"));
}

TEST_CASE("geo: emits an RFC 5870 uri for whatever the device has registered") {
    const std::string url = build_url(NavProvider::GeoUri, bay());
    CHECK(contains(url, "geo:52.3675431,4.8836219"));
    CHECK(contains(url, "Prinsengracht"));
}

// ---------------------------------------------------------------------------
// Refusal
// ---------------------------------------------------------------------------
TEST_CASE("invalid targets produce no link at all") {
    NavTarget nowhere;
    nowhere.lat = 0.0;
    nowhere.lon = 0.0;
    CHECK(nowhere.valid());  // (0,0) is a real coordinate, just not a Dutch one

    NavTarget impossible;
    impossible.lat = 91.0;
    impossible.lon = 4.88;
    CHECK(!impossible.valid());
    CHECK(build_url(NavProvider::GoogleMaps, impossible).empty());
    CHECK(build_links(impossible).empty());
}

TEST_CASE("a not-a-number coordinate is refused rather than printed") {
    NavTarget broken;
    broken.lat = std::nan("");
    broken.lon = 4.88;
    CHECK(!broken.valid());
    CHECK(build_url(NavProvider::GoogleMaps, broken).empty());
}

TEST_CASE("an origin outside the world is treated as absent, not as an error") {
    NavOrigin bad;
    bad.lat = 999.0;
    bad.lon = 4.9;
    bad.present = true;
    CHECK(!bad.valid());

    // The destination is still perfectly usable, so the link must still be produced.
    const std::string url = build_url(NavProvider::GoogleMaps, bay(), bad);
    CHECK(contains(url, "destination=52.3675431,4.8836219"));
    CHECK(!contains(url, "origin="));
}

// ---------------------------------------------------------------------------
// The full set
// ---------------------------------------------------------------------------
TEST_CASE("every provider is offered and each carries the same destination") {
    const auto links = build_links(bay());
    CHECK_EQ(links.size(), 6u);
    for (const auto& link : links) {
        CHECK(!link.provider.empty());
        CHECK(!link.display_name.empty());
        CHECK(contains(link.url, "52.3675431"));
    }
}

TEST_CASE("an entrance is flagged so an interface can say which point it is") {
    NavTarget garage;
    garage.lat = 52.3702;
    garage.lon = 4.8952;
    garage.is_entrance = true;
    CHECK(garage.is_entrance);
    CHECK(!bay().is_entrance);
}

PF_TEST_MAIN()
