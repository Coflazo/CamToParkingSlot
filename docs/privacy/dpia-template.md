# Data Protection Impact Assessment — template

**Status: a template, not a completed assessment.** It is filled in before any camera
watches a public street, and reviewed by someone qualified to sign it off. Shipping this
document as though it were the assessment would be worse than having no document.

---

## Why this is required rather than advisable

Camera images of a public street are personal data whenever a person or a vehicle can be
identified in them, and identifiability does not depend on whether anyone tries. The
GDPR's household exemption — the clause that lets you film your own garden — expressly
does **not** cover monitoring publicly accessible space, so it offers no shelter here.

The Dutch supervisory authority lists large-scale systematic monitoring of publicly
accessible areas as a scenario that requires a DPIA before processing begins. A parking
system watching kerbs across a city is that scenario, whatever the intent.

**None of this depends on whether the project earns money.** There is no non-commercial
exemption in the GDPR. A hobby deployment and a funded service face the same obligation,
and a court would treat "we weren't monetising it" as irrelevant rather than mitigating.

---

## 1. Controller and processors

| Field | Value |
|---|---|
| Data controller | *(the entity deciding why and how the processing happens)* |
| Contact for data subjects | |
| Data protection officer | *(if one is required or appointed)* |
| Processors engaged | *(hosting, camera owners, any third party touching frames)* |
| Processing agreements in place | *(references, dates)* |
| Joint controllers | *(where a municipality or operator shares the decisions)* |

A camera owner who lets you process their feed is very often a **joint controller**
rather than merely a source, because they retain decisions about the camera itself.
Getting that classification wrong changes who is answerable, so record the reasoning.

---

## 2. What is processed

| Category | Present? | Notes |
|---|---|---|
| Video frames of public space | | Held in memory only; released immediately after processing |
| Faces | **Not extracted** | No facial recognition anywhere in the system |
| Licence plates | **Not extracted** | No plate recognition anywhere in the system |
| Vehicle appearance | **Not retained** | Detections exist within one frame and are discarded |
| Occupancy state | Yes | The output: state, geometry, confidence, timestamp |
| Camera identity and location | Yes | Registry metadata |
| User account data | Yes | Email, password hash, vehicle dimensions |
| Licence plate entered by a user | **Discarded after lookup** | Only dimensions are kept, unless the user opts in |
| Destination history | **Opt-in, off by default** | Maps where somebody goes and when |

The rows marked *not extracted* are architectural rather than aspirational. The
publisher emits only state, geometry, confidence and timestamps; there is no code path
that writes a frame, a crop or an appearance descriptor to storage.

**Restate this section against the deployment as built.** A design that discards frames
is not the same as a deployment that does, and only the second is what gets assessed.

---

## 3. Purpose and lawful basis

| Question | Answer |
|---|---|
| Purpose | |
| Lawful basis relied on | |
| If legitimate interests: the balancing test | *(attach)* |
| Is the original camera purpose compatible with this one? | |
| Would data subjects reasonably expect this? | |

A camera installed for public-order supervision was not installed to count parking
spaces. Purpose compatibility under Article 6(4) is a real question with a real answer,
and it has to be argued rather than assumed.

---

## 4. Necessity and proportionality

| Question | Answer |
|---|---|
| Could the purpose be achieved without cameras? | |
| Why is this sampling rate the minimum? | Default 0.125 fps — parking changes over minutes |
| Why is this field of view the minimum? | |
| Are privacy masks applied to pavements, windows, doorways? | |
| What is the retention period, and why that one? | |
| How is the volume of processing limited? | |

Slow sampling is a privacy control, not only a performance one: it directly reduces how
much imagery of a public street exists at any moment.

---

## 5. Risks to data subjects

| Risk | Likelihood | Severity | Mitigation | Residual |
|---|---|---|---|---|
| Individuals identifiable in frames | | | Frames never persisted; no recognition of any kind | |
| Vehicle movements inferable over time | | | Only occupancy state stored, never vehicle identity | |
| Function creep toward surveillance | | | Registry gate; no plate or face capability exists to enable | |
| Camera feed accessed without authorisation | | | Credentials stored separately from user data and encrypted | |
| Users' destination history revealing patterns of life | | | Opt-in, off by default, deletable | |
| A wrong recommendation causing a fine or damage | | | Fit floors; UNVERIFIED shown rather than guessed | |

---

## 6. Technical and organisational measures

Implemented in this codebase:

- [x] No facial recognition, plate recognition or demographic analysis anywhere
- [x] Frames held in memory and released immediately after processing
- [x] Only occupancy, geometry, confidence and timestamps published
- [x] Camera registry gate: a worker refuses an uncleared feed as a hard stop
- [x] Owner attestation requires a reference to an actual permission
- [x] Two-tier permission model: research on one machine ≠ running a service
- [x] Argon2id password hashing, short-lived JWTs
- [x] Account deletion cascades to vehicles
- [x] Destination history opt-in and off by default
- [x] Licence plate discarded after the RDW lookup
- [x] Per-source licence registry so terms are recorded rather than assumed
- [x] robots.txt honoured; no anti-bot evasion

To be completed per deployment:

- [ ] Privacy masks configured for each camera's field of view
- [ ] Retention schedule agreed and enforced
- [ ] Public privacy notice published where required
- [ ] Data-subject request procedure in place and tested
- [ ] Camera credentials in a secrets manager, not in the database
- [ ] EU-hosted infrastructure confirmed
- [ ] Access logging and review cadence agreed
- [ ] Breach notification procedure tested

---

## 7. Consultation

| Party | Consulted | Date | Outcome |
|---|---|---|---|
| Data protection officer | | | |
| Camera owners / operators | | | |
| Municipality | | | |
| Supervisory authority (if residual risk stays high) | | | |
| Representatives of data subjects | | | |

Article 36 requires prior consultation with the supervisory authority when a DPIA shows
high residual risk that cannot be mitigated. That is a decision to record, not to skip.

---

## 8. Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| Data protection officer | | | |
| Controller representative | | | |
| Technical lead | | | |
| Legal review (Dutch privacy counsel) | | | |

**Review triggers:** any new camera, any change of field of view, any new purpose, any
change to what is published, any change of processor — and at minimum annually.

---

## References

- [Dutch DPA — DPIA guidance](https://www.autoriteitpersoonsgegevens.nl/en/themes/basic-gdpr/gdpr-in-practice/data-protection-impact-assessment-dpia)
- [EDPB Guidelines 3/2019 on processing personal data through video devices](https://www.edpb.europa.eu/documents/guideline/guidelines-32019-on-processing-of-personal-data-through-video-devices_en)
- [Amsterdam camera surveillance policy](https://www.amsterdam.nl/privacy/cameratoezicht/)
- [Police "Camera in Beeld"](https://www.politie.nl/onderwerpen/camera-in-beeld.html) — a registry for requesting evidence after a crime, not a live feed API
