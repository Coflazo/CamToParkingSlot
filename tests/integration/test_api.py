"""API integration tests.

These run the real application against a temporary database, so they exercise routing,
serialisation, auth and the evidence layer together. Network-dependent paths (RDW plate
lookup, geocoding through PDOK) are marked so the suite still runs offline.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from parkfit.storage.models import ParkingFacility, PointOfInterest


@pytest.fixture
def client(session):
    """A test client sharing the isolated database."""
    from parkfit.api.app import app
    from parkfit.services.candidate_index import get_candidate_index

    # The index is process-wide and may hold rows from another test.
    get_candidate_index().invalidate()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def registered(client):
    """An account with a bearer token."""
    response = client.post(
        "/v1/auth/register",
        json={"email": "driver@example.com", "password": "a-sufficiently-long-password"},
    )
    assert response.status_code == 201
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def seeded_destination(session, seeded_facilities):
    """A point of interest to search for, plus facilities near it."""
    session.add(
        PointOfInterest(
            source_name="OpenStreetMap", external_id="way/1", name="Rembrandthuis",
            normalised_name="rembrandthuis", category="museum",
            lat=52.36937, lon=4.90125, city="Amsterdam", importance=0.95,
        )
    )
    session.commit()
    from parkfit.services.candidate_index import get_candidate_index

    get_candidate_index().invalidate()
    return seeded_facilities


class TestHealth:
    def test_health_reports_what_is_loaded(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "native_module" in body
        assert body["database"] == "sqlite"
        # Reporting counts rather than a bare "ok" is the point: a service whose road
        # graph failed to load should not claim to be healthy.
        assert "facilities" in body and "bays" in body

    def test_index_carries_source_attribution(self, client):
        """OpenStreetMap attribution is a licence obligation, not a courtesy."""
        body = client.get("/").json()
        assert any("OpenStreetMap" in line for line in body["attribution"])


class TestAuth:
    def test_register_then_login(self, client):
        assert client.post(
            "/v1/auth/register",
            json={"email": "a@example.com", "password": "a-sufficiently-long-password"},
        ).status_code == 201
        response = client.post(
            "/v1/auth/login",
            json={"email": "a@example.com", "password": "a-sufficiently-long-password"},
        )
        assert response.status_code == 200
        assert response.json()["access_token"]

    def test_duplicate_registration_is_rejected(self, client):
        payload = {"email": "b@example.com", "password": "a-sufficiently-long-password"}
        client.post("/v1/auth/register", json=payload)
        assert client.post("/v1/auth/register", json=payload).status_code == 409

    def test_wrong_password_is_rejected(self, client):
        client.post(
            "/v1/auth/register",
            json={"email": "c@example.com", "password": "a-sufficiently-long-password"},
        )
        assert client.post(
            "/v1/auth/login", json={"email": "c@example.com", "password": "wrong-password-here"}
        ).status_code == 401

    def test_an_unknown_email_is_rejected_the_same_way(self, client):
        """Same status and shape as a wrong password, so the response does not reveal
        which addresses are registered."""
        assert client.post(
            "/v1/auth/login",
            json={"email": "nobody@example.com", "password": "a-sufficiently-long-password"},
        ).status_code == 401

    def test_a_short_password_is_refused(self, client):
        assert client.post(
            "/v1/auth/register", json={"email": "d@example.com", "password": "short"}
        ).status_code == 422

    def test_protected_routes_require_a_token(self, client):
        assert client.get("/v1/vehicles").status_code == 401

    def test_an_invalid_token_is_rejected(self, client):
        assert client.get(
            "/v1/vehicles", headers={"Authorization": "Bearer not-a-real-token"}
        ).status_code == 401


class TestVehicles:
    def _payload(self, **overrides):
        payload = {
            "nickname": "Polo", "length_cm": 405.3, "body_width_cm": 175.1,
            "width_with_mirrors_cm": 194.0, "height_cm": 145.1, "weight_kg": 1105.0,
        }
        payload.update(overrides)
        return payload

    def test_create_and_list(self, client, registered):
        created = client.post("/v1/vehicles", headers=registered, json=self._payload())
        assert created.status_code == 201
        assert created.json()["nickname"] == "Polo"

        listed = client.get("/v1/vehicles", headers=registered).json()
        assert len(listed) == 1

    def test_a_missing_mirror_width_is_inferred_and_flagged(self, client, registered):
        body = client.post(
            "/v1/vehicles", headers=registered, json=self._payload(width_with_mirrors_cm=0.0)
        ).json()
        assert body["width_with_mirrors_cm"] == pytest.approx(175.1 + 36.0)
        # Inferred, so it must not be recorded as confirmed.
        assert body["width_confirmed"] is False

    def test_accessory_height_defaults_to_body_height(self, client, registered):
        body = client.post("/v1/vehicles", headers=registered, json=self._payload()).json()
        assert body["height_with_accessories_cm"] == pytest.approx(145.1)

    def test_update_confirms_the_dimension(self, client, registered):
        vehicle_id = client.post(
            "/v1/vehicles", headers=registered, json=self._payload()
        ).json()["id"]
        updated = client.patch(
            f"/v1/vehicles/{vehicle_id}", headers=registered, json={"height_cm": 152.0}
        ).json()
        assert updated["height_cm"] == pytest.approx(152.0)
        assert updated["height_confirmed"] is True

    def test_another_users_vehicle_is_not_found(self, client, registered):
        vehicle_id = client.post(
            "/v1/vehicles", headers=registered, json=self._payload()
        ).json()["id"]

        client.post(
            "/v1/auth/register",
            json={"email": "other@example.com", "password": "a-sufficiently-long-password"},
        )
        other = client.post(
            "/v1/auth/login",
            json={"email": "other@example.com", "password": "a-sufficiently-long-password"},
        ).json()["access_token"]

        # 404 rather than 403: a different status would confirm the id exists.
        assert client.patch(
            f"/v1/vehicles/{vehicle_id}",
            headers={"Authorization": f"Bearer {other}"},
            json={"height_cm": 200.0},
        ).status_code == 404

    def test_implausible_dimensions_are_refused(self, client, registered):
        assert client.post(
            "/v1/vehicles", headers=registered, json=self._payload(length_cm=-5.0)
        ).status_code == 422
        assert client.post(
            "/v1/vehicles", headers=registered, json=self._payload(height_cm=9999.0)
        ).status_code == 422


class TestSearch:
    def test_search_returns_ranked_results_with_evidence(
        self, client, registered, seeded_destination
    ):
        response = client.post(
            "/v1/searches",
            headers=registered,
            json={
                "destination": "Rembrandthuis",
                "origin_lat": 52.3789, "origin_lon": 4.9002,
                "expected_duration_minutes": 120,
                "preferences": {"max_walk_minutes": 25, "include_on_street": False},
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["destination"]["label"] == "Rembrandthuis"
        assert body["results"]

        for result in body["results"]:
            # Every claim carries its provenance. That is the product promise.
            assert result["evidence"]["source"]
            assert result["evidence"]["freshness"]
            assert result["evidence"]["confidence_label"]
            assert result["fit"]["verdict"] in {
                "FITS", "TIGHT_FIT", "UNVERIFIED", "DOES_NOT_FIT"
            }
            assert result["fit"]["explanation"]

    def test_results_are_ordered_by_generalised_cost(
        self, client, registered, seeded_destination
    ):
        body = client.post(
            "/v1/searches",
            headers=registered,
            json={
                "destination": "Rembrandthuis", "origin_lat": 52.3789, "origin_lon": 4.9002,
                "preferences": {"max_walk_minutes": 30, "include_on_street": False},
            },
        ).json()
        costs = [r["generalised_cost_eur"] for r in body["results"]]
        assert costs == sorted(costs)

    def test_a_van_is_excluded_from_a_low_garage(self, client, registered, seeded_destination):
        van = client.post(
            "/v1/vehicles", headers=registered,
            json={
                "nickname": "Transporter", "length_cm": 590.0, "body_width_cm": 190.4,
                "width_with_mirrors_cm": 246.0, "height_cm": 199.0,
                "height_with_accessories_cm": 232.0, "weight_kg": 2000.0,
            },
        ).json()["id"]

        body = client.post(
            "/v1/searches", headers=registered,
            json={
                "destination": "Rembrandthuis", "vehicle_id": van,
                "origin_lat": 52.3789, "origin_lon": 4.9002,
                "preferences": {"max_walk_minutes": 30, "include_on_street": False},
            },
        ).json()

        # Garage Laag publishes a 180 cm barrier; a 232 cm van must never appear.
        names = [r["name"] for r in body["results"]]
        assert not any("Laag" in name for name in names)
        assert body["rejected_fit"] >= 1

    def test_an_unknown_destination_reports_it_rather_than_guessing(self, client, registered):
        body = client.post(
            "/v1/searches", headers=registered,
            json={"destination": "zzzzqqqxx not a real place at all"},
        ).json()
        assert body["results"] == [] or body["warnings"]

    def test_anonymous_search_is_allowed(self, client, seeded_destination):
        """A driver should not need an account to find out where they can park."""
        response = client.post(
            "/v1/searches",
            json={
                "destination": "Rembrandthuis", "origin_lat": 52.3789, "origin_lon": 4.9002,
                "preferences": {"max_walk_minutes": 30, "include_on_street": False},
            },
        )
        assert response.status_code == 201

    def test_searching_with_a_saved_vehicle_requires_a_token(self, client, seeded_destination):
        assert client.post(
            "/v1/searches", json={"destination": "Rembrandthuis", "vehicle_id": 1}
        ).status_code == 401

    def test_an_unknown_vehicle_is_rejected(self, client, registered, seeded_destination):
        assert client.post(
            "/v1/searches",
            headers=registered,
            json={"destination": "Rembrandthuis", "vehicle_id": 99999},
        ).status_code == 404


class TestGeocode:
    def test_local_points_of_interest_are_searched_first(self, client, seeded_destination):
        """The whole reason the geocoder is hybrid: the official Dutch geocoder returns
        nothing for "Rembrandthuis" because it indexes addresses, not places."""
        body = client.get("/v1/geocode", params={"q": "Rembrandthuis"}).json()
        assert body["results"]
        assert body["results"][0]["source"] == "OpenStreetMap"
        assert body["results"][0]["confidence"] > 0.7

    def test_a_descriptive_query_still_matches(self, client, seeded_destination):
        body = client.get("/v1/geocode", params={"q": "Rembrandt House Museum"}).json()
        assert body["results"]
        assert "Rembrandt" in body["results"][0]["label"]

    def test_a_short_query_is_refused(self, client):
        assert client.get("/v1/geocode", params={"q": "a"}).status_code == 422


class TestObservations:
    def test_a_user_confirmation_is_recorded_as_an_observation(
        self, client, session, seeded_facilities
    ):
        facility_id = seeded_facilities[0].id
        response = client.post(
            "/v1/observations/user-confirmation",
            json={"target_kind": "facility", "target_id": facility_id,
                  "outcome": "was_occupied"},
        )
        assert response.status_code == 201

        from sqlalchemy import select

        from parkfit.storage.models import AvailabilityObservation, EvidenceSource

        session.expire_all()
        rows = session.execute(
            select(AvailabilityObservation).where(
                AvailabilityObservation.target_id == facility_id
            )
        ).scalars().all()
        assert rows
        # Recorded at user priority: enough to correct a stale feed, not enough to
        # override an operator counting spaces at its own barrier.
        assert rows[0].evidence_source == int(EvidenceSource.USER_CONFIRMATION)

    def test_an_invalid_outcome_is_refused(self, client):
        assert client.post(
            "/v1/observations/user-confirmation",
            json={"target_kind": "facility", "target_id": 1, "outcome": "maybe"},
        ).status_code == 422


class TestFacilityDetail:
    def test_detail_includes_attribution(self, client, session, seeded_facilities):
        from parkfit.storage.models import SourceLicence

        session.add(
            SourceLicence(
                source_name="RDW-NPR", dataset_url="https://opendata.rdw.nl/",
                licence="CC0-1.0", attribution_text="Data: RDW",
            )
        )
        session.commit()
        body = client.get(f"/v1/parking/{seeded_facilities[0].id}").json()
        assert body["name"] == "Garage The Bank (Amsterdam)"
        assert body["max_vehicle_height_cm"] == pytest.approx(210.0)
        assert body["attribution"] == "Data: RDW"

    def test_an_unknown_facility_is_404(self, client):
        assert client.get("/v1/parking/999999").status_code == 404


class TestAvailabilityStream:
    def test_invalid_targets_are_refused(self, client):
        assert client.get(
            "/v1/availability/stream", params={"targets": "nonsense"}
        ).status_code == 400
