"""Camera registry: which feeds this deployment may process, and why.

There is no nationwide network of curb-facing Dutch camera feeds that software can
simply consume. Amsterdam's public-space cameras are watched by municipal supervisors
and only police may review recordings; the police "Camera in Beeld" system is a registry
for requesting evidence after a crime, not a live API; Rijkswaterstaat publishes a
couple of dozen road-facing cameras that point at motorway lanes rather than parking.

So this module does not discover cameras. It records decisions about them. Every feed
carries a permission status, and a worker cannot open one whose status this deployment
does not accept.

The gate is deliberately two-sided:

* **Research on your own machine** (``PARKFIT_ENVIRONMENT=dev``) accepts a feed whose
  host permits crawling and whose terms show no prohibition. That is enough to build and
  test against real imagery.
* **Running a service** (``prod``) accepts only an explicit authorisation or an owner
  attestation, because at that point you are processing other people's cameras at scale
  and "the robots file allowed it" is not a licence.

What the gate never does is guess. A feed that has not been assessed is ``UNVERIFIED``
and does not run, in either mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from parkfit.config import Environment, Settings, get_settings
from parkfit.storage.models import CameraPermission, CameraSource, FrameHealth

log = logging.getLogger(__name__)

#: Statuses that permit automated processing, by environment.
ACCEPTED_IN_PROD = frozenset({CameraPermission.AUTHORISED, CameraPermission.OWNER_ATTESTED})
ACCEPTED_IN_DEV = ACCEPTED_IN_PROD | {CameraPermission.ROBOTS_OK}


@dataclass(frozen=True)
class PermissionDecision:
    """Whether a feed may be processed, and the reason either way."""

    allowed: bool
    status: CameraPermission
    reason: str
    environment: str

    def explain(self) -> str:
        verb = "may" if self.allowed else "may not"
        return f"{self.status.value}: {verb} be processed in {self.environment} - {self.reason}"


def evaluate_permission(
    camera: CameraSource, settings: Settings | None = None
) -> PermissionDecision:
    """Decide whether this camera may be opened by a worker."""
    settings = settings or get_settings()
    environment = settings.environment.value
    try:
        status = CameraPermission(camera.permission_status)
    except ValueError:
        status = CameraPermission.UNVERIFIED

    if status is CameraPermission.BLOCKED:
        return PermissionDecision(
            False, status, "the host or its terms forbid automated access", environment
        )

    if status is CameraPermission.UNVERIFIED:
        return PermissionDecision(
            False,
            status,
            "not assessed yet; run the source auditor or record an authorisation",
            environment,
        )

    accepted = (
        ACCEPTED_IN_PROD
        if settings.environment is Environment.PROD or not settings.camera_allow_robots_ok
        else ACCEPTED_IN_DEV
    )

    if status not in accepted:
        return PermissionDecision(
            False,
            status,
            "crawling is permitted but the terms are unverified, which is enough for "
            "local research and not enough to run a service",
            environment,
        )

    # An explicit false beats a permissive status. Someone took the trouble to record
    # that automated processing is not allowed, and that outranks a robots file.
    if camera.automated_processing_allowed is False:
        return PermissionDecision(
            False,
            status,
            "automated processing is explicitly recorded as not allowed",
            environment,
        )

    return PermissionDecision(True, status, "cleared for automated processing", environment)


class CameraRegistry:
    """Reads and writes the camera registry."""

    def __init__(self, session: Session, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()

    # -- reading ------------------------------------------------------------
    def get(self, camera_id: str) -> CameraSource | None:
        return self.session.execute(
            select(CameraSource).where(CameraSource.camera_id == camera_id)
        ).scalar_one_or_none()

    def all(self) -> list[CameraSource]:
        return list(self.session.execute(select(CameraSource)).scalars())

    def processable(self) -> list[tuple[CameraSource, PermissionDecision]]:
        """Every camera, paired with whether it may run here."""
        return [(c, evaluate_permission(c, self.settings)) for c in self.all()]

    def runnable(self) -> list[CameraSource]:
        return [c for c, d in self.processable() if d.allowed and c.enabled]

    # -- writing ------------------------------------------------------------
    def register(
        self,
        camera_id: str,
        *,
        stream_url: str,
        stream_type: str,
        owner: str | None = None,
        operator: str | None = None,
        public_page_url: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        heading_deg: float | None = None,
        fixed_mount: bool = True,
        permission_status: str = CameraPermission.UNVERIFIED.value,
        licence_url: str | None = None,
        agreement_reference: str | None = None,
        terms_url: str | None = None,
        robots_allowed: bool | None = None,
        automated_processing_allowed: bool | None = None,
        sample_interval_s: float | None = None,
        notes: str | None = None,
    ) -> CameraSource:
        camera = self.get(camera_id)
        if camera is None:
            camera = CameraSource(camera_id=camera_id, source_name="registry")
            self.session.add(camera)

        camera.stream_url = stream_url
        camera.stream_type = stream_type
        camera.owner = owner
        camera.operator = operator
        camera.public_page_url = public_page_url
        camera.lat = lat
        camera.lon = lon
        camera.heading_deg = heading_deg
        camera.fixed_mount = fixed_mount
        camera.permission_status = permission_status
        camera.licence_url = licence_url
        camera.agreement_reference = agreement_reference
        camera.terms_url = terms_url
        camera.robots_allowed = robots_allowed
        camera.automated_processing_allowed = automated_processing_allowed
        camera.sample_interval_s = sample_interval_s or self.settings.camera_frame_interval_s
        camera.notes = notes
        camera.fetched_at = datetime.now(UTC)

        # A newly-registered camera is never enabled by default. Enabling is a separate,
        # deliberate act, so a bulk import of audit results cannot start processing feeds
        # nobody has looked at.
        if camera.id is None:
            camera.enabled = False
            camera.technical_status = FrameHealth.OFFLINE.value

        self.session.flush()
        return camera

    def attest_ownership(
        self, camera_id: str, *, agreement_reference: str, reviewer: str | None = None
    ) -> CameraSource:
        """Record that the operator holds the rights to process this feed.

        This is the status production accepts, and it is an assertion by a person, not a
        result the software can derive. The agreement reference is required so the claim
        points at something -- a contract, a licence, a written permission -- rather than
        being an unsupported flag in a database.
        """
        camera = self.get(camera_id)
        if camera is None:
            raise KeyError(f"unknown camera: {camera_id}")
        if not agreement_reference.strip():
            raise ValueError("an agreement reference is required to attest ownership")

        camera.permission_status = CameraPermission.OWNER_ATTESTED.value
        camera.agreement_reference = agreement_reference.strip()
        camera.automated_processing_allowed = True
        camera.last_legal_review = datetime.now(UTC)
        if reviewer:
            camera.notes = f"{camera.notes or ''}\nattested by {reviewer}".strip()
        self.session.flush()
        return camera

    def block(self, camera_id: str, reason: str) -> CameraSource:
        camera = self.get(camera_id)
        if camera is None:
            raise KeyError(f"unknown camera: {camera_id}")
        camera.permission_status = CameraPermission.BLOCKED.value
        camera.automated_processing_allowed = False
        camera.enabled = False
        camera.notes = f"{camera.notes or ''}\nblocked: {reason}".strip()
        self.session.flush()
        return camera

    def set_enabled(self, camera_id: str, enabled: bool) -> CameraSource:
        """Enable or disable a camera, refusing to enable one that may not run."""
        camera = self.get(camera_id)
        if camera is None:
            raise KeyError(f"unknown camera: {camera_id}")
        if enabled:
            decision = evaluate_permission(camera, self.settings)
            if not decision.allowed:
                raise PermissionError(decision.explain())
        camera.enabled = enabled
        self.session.flush()
        return camera

    def record_health(self, camera_id: str, health: str) -> None:
        camera = self.get(camera_id)
        if camera is None:
            return
        camera.technical_status = health
        camera.last_frame_at = datetime.now(UTC)
        self.session.flush()

    # -- reporting ----------------------------------------------------------
    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for camera in self.all():
            counts[camera.permission_status] = counts.get(camera.permission_status, 0) + 1
        return counts
