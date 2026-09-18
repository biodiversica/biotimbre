"""Deciding what an incoming bundle should become in *this* library.

:mod:`biotimbre.io.bundle` knows the file format and nothing about a library.
:class:`~biotimbre.service.LibraryService` owns transactions and the
filesystem. Between them sits the question this module answers: given a bundle
someone sent and the library that is about to receive it, what happens to each
entry — and why?

That question is worth its own place because it is the part a person has to
agree with before anything is written. Every decision here is made by
:func:`plan_import`, which touches nothing: the service runs it to show a
preview, then runs it again to import, so the preview cannot describe one
outcome while the import performs another.

Four kinds of disagreement come up, and the library has a position on each:

**The same recording, already here.** Matched on ``audio_sha256``, the digest
of the stored file. If it is filed under the same biotimbre it is the same
sample and is skipped. Under a *different* biotimbre it is the same recording
with two identities, which is a real disagreement between two libraries rather
than a duplicate to quietly drop, so it is reported and skipped by default.

**A species that already has a biotimbre here.** Two people who each catalogued
*Guyalna bonaerensis* have two UUIDs for one sound-being, and the library's rule
is one species, one biotimbre (see
:meth:`~biotimbre.service.LibraryService._join_species_biotimbre`). The default
is to join the local one, keeping search and grouping coherent; the sender's id
is reported so nothing is lost silently.

**Embeddings from a model this library does not have, or has differently.**
Vectors are comparable only when the whole preprocessing recipe matches, so a
fingerprint disagreement rejects them and they are recomputed. Accepting them
would corrupt search results in a way nothing downstream could detect.

**A bundle whose sender you do not trust.** ``trust_embeddings=False`` throws
every vector away and recomputes from the audio, which is the only part of a
bundle that can be checked against itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

from biotimbre.db.repository import Repository
from biotimbre.domain.errors import ValidationError
from biotimbre.domain.item import ItemDraft, LibraryItem, is_species_name
from biotimbre.io.bundle import Bundle, BundleEntry, ModelFingerprint, Provenance

#: What to do when a bundle's audio is already here under another biotimbre.
DuplicatePolicy = Literal["skip", "add"]

#: What to do when an incoming species already has a biotimbre in this library.
SpeciesPolicy = Literal["join_local", "keep_incoming"]

#: What an entry will become, or became.
Action = Literal["add", "skip", "failed"]


def fingerprint_from_row(row: Mapping) -> ModelFingerprint:
    """Build a fingerprint from a ``models`` table row.

    Args:
        row: A row from :meth:`~biotimbre.db.repository.Repository.get_model`
            or :meth:`~biotimbre.db.repository.Repository.list_models`.

    Returns:
        The model's :class:`~biotimbre.io.bundle.ModelFingerprint`.
    """
    return ModelFingerprint(
        model_id=str(row["model_id"]),
        source=str(row["source"]),
        filename=str(row["filename"]),
        revision=None if row["revision"] is None else str(row["revision"]),
        sample_rate=int(row["sample_rate"]),
        window_duration_s=float(row["window_duration_s"]),
        window_overlap=float(row["window_overlap"]),
        embedding_dim=int(row["embedding_dim"]),
    )


def local_fingerprints(repo: Repository) -> dict[str, ModelFingerprint]:
    """Fingerprints for every model this library has registered.

    Args:
        repo: The library's repository.

    Returns:
        Fingerprint per model id.
    """
    return {
        str(row["model_id"]): fingerprint_from_row(row) for row in repo.list_models()
    }


@dataclass(frozen=True)
class ImportPolicy:
    """The choices an import makes on the user's behalf.

    Attributes:
        trust_embeddings: Use the bundle's vectors where their model
            fingerprint matches this library's. False recomputes everything
            from the audio, which is the only part of a bundle that verifies
            itself.
        compute_embeddings: Compute whatever the bundle did not supply, so an
            imported item is searchable on every active model. False leaves the
            gaps for :meth:`~biotimbre.service.LibraryService.backfill`, which
            is what a bulk import on a machine without the model files wants.
        duplicate_audio: What to do when this library already holds the same
            recording under a *different* biotimbre.
        species_conflict: What to do when an incoming species already has a
            biotimbre here.
    """

    trust_embeddings: bool = True
    compute_embeddings: bool = True
    duplicate_audio: DuplicatePolicy = "skip"
    species_conflict: SpeciesPolicy = "join_local"

    def __post_init__(self) -> None:
        if self.duplicate_audio not in ("skip", "add"):
            raise ValidationError(
                f"duplicate_audio must be 'skip' or 'add', got "
                f"{self.duplicate_audio!r}.",
                "duplicate_audio",
            )
        if self.species_conflict not in ("join_local", "keep_incoming"):
            raise ValidationError(
                f"species_conflict must be 'join_local' or 'keep_incoming', got "
                f"{self.species_conflict!r}.",
                "species_conflict",
            )


@dataclass(frozen=True)
class EntryPlan:
    """What one bundle entry will become in this library.

    Attributes:
        index: The entry's 1-based position in the bundle.
        entry: The entry itself, for its audio and its vectors.
        draft: The metadata as it will be stored — ``biotimbre_id`` already
            resolved, which is the one field an import may change.
        incoming_biotimbre_id: The id the sender filed it under. Differs from
            ``draft.biotimbre_id`` exactly when a species conflict was resolved
            in favour of a local biotimbre.
        action: ``"add"`` or ``"skip"``.
        reason: Why it is skipped; ``None`` when it is not.
        reusable: Models whose vectors will be taken from the bundle.
        rejected: Model id to the reason its vectors were refused. Those
            models are recomputed when ``compute_embeddings`` is set.
        warnings: Advisories a person should read before agreeing.
    """

    index: int
    entry: BundleEntry
    draft: ItemDraft
    incoming_biotimbre_id: str
    action: Action
    reason: str | None = None
    reusable: tuple[str, ...] = ()
    rejected: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def biotimbre_id(self) -> str:
        """The biotimbre this entry will actually be filed under."""
        assert self.draft.biotimbre_id is not None
        return self.draft.biotimbre_id

    @property
    def rebased(self) -> bool:
        """Whether the entry is being moved off the sender's biotimbre."""
        return self.biotimbre_id != self.incoming_biotimbre_id

    @property
    def provenance(self) -> Provenance:
        """Where the entry came from."""
        return self.entry.provenance

    @property
    def reusable_rows(self) -> dict[str, list]:
        """The vectors to store as-is, ready for the repository."""
        return {model_id: self.entry.embeddings[model_id] for model_id in self.reusable}


@dataclass(frozen=True)
class BundlePreview:
    """What importing a bundle would do, without doing any of it.

    Attributes:
        bundle: The opened bundle.
        policy: The policy the plans were made under.
        plans: One plan per entry, in bundle order.
    """

    bundle: Bundle
    policy: ImportPolicy
    plans: tuple[EntryPlan, ...]

    def __len__(self) -> int:
        return len(self.plans)

    @property
    def to_add(self) -> tuple[EntryPlan, ...]:
        """Plans that will store something."""
        return tuple(plan for plan in self.plans if plan.action == "add")

    @property
    def to_skip(self) -> tuple[EntryPlan, ...]:
        """Plans that will not."""
        return tuple(plan for plan in self.plans if plan.action == "skip")

    @property
    def has_warnings(self) -> bool:
        """Whether anything here deserves a person's attention before importing."""
        return any(plan.warnings for plan in self.plans)


@dataclass(frozen=True)
class ImportOutcome:
    """What actually happened to one entry.

    Attributes:
        plan: The plan that was carried out.
        item: The stored item, or ``None`` if nothing was stored.
        window_counts: Windows written per model, reused and recomputed alike.
        action: ``"add"``, ``"skip"``, or ``"failed"``.
        reason: Why it was skipped, or what went wrong.
        warnings: Advisories from planning, plus any the ingest itself raised —
            a Nyquist mismatch, say. A skipped entry has none: the reason it
            was skipped is the whole story.
    """

    plan: EntryPlan
    item: LibraryItem | None
    window_counts: dict[str, int] = field(default_factory=dict)
    action: Action = "add"
    reason: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def biotimbre_id(self) -> str:
        """The biotimbre the entry was filed under."""
        return self.plan.biotimbre_id

    @property
    def provenance(self) -> Provenance:
        """Where the entry came from."""
        return self.plan.provenance


@dataclass(frozen=True)
class ImportReport:
    """The result of importing one bundle.

    An entry that cannot be *stored* does not abort the rest, so this reports
    per entry rather than raising. An entry that cannot be *read* is a
    different matter: :func:`~biotimbre.io.bundle.read_bundle` rejects the
    whole file, so a report only ever describes a bundle that was sound.

    Attributes:
        bundle: The bundle that was imported.
        policy: The policy it was imported under.
        outcomes: One per entry, in bundle order.
    """

    bundle: Bundle
    policy: ImportPolicy
    outcomes: tuple[ImportOutcome, ...]

    def __len__(self) -> int:
        return len(self.outcomes)

    @property
    def added(self) -> tuple[ImportOutcome, ...]:
        """Entries that became items."""
        return tuple(o for o in self.outcomes if o.action == "add")

    @property
    def skipped(self) -> tuple[ImportOutcome, ...]:
        """Entries deliberately not stored."""
        return tuple(o for o in self.outcomes if o.action == "skip")

    @property
    def failed(self) -> tuple[ImportOutcome, ...]:
        """Entries that could not be stored."""
        return tuple(o for o in self.outcomes if o.action == "failed")

    @property
    def items(self) -> tuple[LibraryItem, ...]:
        """The items this import created."""
        return tuple(o.item for o in self.added if o.item is not None)


def plan_import(
    repo: Repository, bundle: Bundle, policy: ImportPolicy | None = None
) -> BundlePreview:
    """Decide what each of a bundle's entries should become.

    Reads the library but writes nothing, so the answer can be shown to a person
    and then produced again, unchanged, when they agree.

    Entries are planned in order against a running record of what earlier
    entries settled, so a bundle that files one species under two biotimbres
    converges on one here rather than reproducing the sender's disagreement.

    Args:
        repo: The receiving library's repository.
        bundle: The opened bundle.
        policy: The choices to plan under; defaults apply if omitted.

    Returns:
        A :class:`BundlePreview`.
    """
    resolved_policy = policy or ImportPolicy()
    local_models = local_fingerprints(repo)
    settled: dict[str, str] = {}

    plans = tuple(
        _plan_entry(repo, bundle, entry, index, resolved_policy, local_models, settled)
        for index, entry in enumerate(bundle.entries, start=1)
    )
    return BundlePreview(bundle=bundle, policy=resolved_policy, plans=plans)


def _plan_entry(
    repo: Repository,
    bundle: Bundle,
    entry: BundleEntry,
    index: int,
    policy: ImportPolicy,
    local_models: Mapping[str, ModelFingerprint],
    settled: dict[str, str],
) -> EntryPlan:
    """Decide one entry's fate. See :func:`plan_import`."""
    from dataclasses import replace

    incoming = entry.draft.biotimbre_id
    assert incoming is not None  # read_bundle refuses an entry without one

    target, warnings = _resolve_biotimbre(repo, entry.draft, incoming, policy, settled)
    reusable, rejected = _plan_embeddings(bundle, entry, policy, local_models)

    action, reason, duplicate_warnings = _check_duplicates(repo, entry, target, policy)
    warnings += duplicate_warnings

    return EntryPlan(
        index=index,
        entry=entry,
        draft=replace(entry.draft, biotimbre_id=target),
        incoming_biotimbre_id=incoming,
        action=action,
        reason=reason,
        reusable=reusable,
        rejected=rejected,
        warnings=warnings,
    )


def _resolve_biotimbre(
    repo: Repository,
    draft: ItemDraft,
    incoming: str,
    policy: ImportPolicy,
    settled: dict[str, str],
) -> tuple[str, tuple[str, ...]]:
    """Choose the biotimbre an entry is filed under.

    The first species tag decides, as it does on the ingest side: a rank
    ("cicadidae") says nothing about which being was recorded, and open
    nomenclature ("guyalna sp.") is precisely an admission that the species is
    unknown.

    Args:
        repo: The receiving library's repository.
        draft: The entry's metadata.
        incoming: The biotimbre the sender filed it under.
        policy: The choices to apply.
        settled: Species this import has already decided, updated in place.

    Returns:
        The biotimbre to use, and any advisories.
    """
    for tag in draft.taxa:
        if not is_species_name(tag):
            continue

        if tag in settled:
            chosen = settled[tag]
            if chosen == incoming:
                return incoming, ()
            return chosen, (
                f"{tag} was filed under {chosen} earlier in this bundle; this "
                f"sample joins it rather than becoming a second identity for "
                f"one species.",
            )

        local = [
            existing
            for existing in repo.biotimbres_with_taxon(tag)
            if existing != incoming
        ]
        if not local:
            settled[tag] = incoming
            return incoming, ()

        if policy.species_conflict == "join_local":
            settled[tag] = local[0]
            return local[0], (
                f"{tag} already has a biotimbre here ({local[0]}); this sample "
                f"joins it rather than adding a second identity for one "
                f"species. The sender filed it under {incoming}.",
            )

        settled[tag] = incoming
        return incoming, (
            f"{tag} is already filed here under {local[0]}, and this sample "
            f"keeps the sender's {incoming}: one species will have two "
            f"biotimbres until they are merged.",
        )

    return incoming, ()


def _check_duplicates(
    repo: Repository, entry: BundleEntry, target: str, policy: ImportPolicy
) -> tuple[Action, str | None, tuple[str, ...]]:
    """Decide whether this library already holds the entry's recording.

    Matched on the stored file's digest, which is what makes the answer
    meaningful across libraries: two copies of one recording hash the same no
    matter what either library called it.

    Args:
        repo: The receiving library's repository.
        entry: The entry being planned.
        target: The biotimbre it would be filed under.
        policy: The choices to apply.

    Returns:
        The action, the reason for a skip, and any advisories.
    """
    existing = repo.find_by_sha256(entry.audio_info.sha256)
    if not existing:
        return "add", None, ()

    same_identity = [item for item in existing if item.biotimbre_id == target]
    if same_identity:
        ids = ", ".join(str(item.item_id) for item in same_identity)
        return (
            "skip",
            f"already in this library as item {ids}, under the same biotimbre.",
            (),
        )

    others = ", ".join(
        f"item {item.item_id} ({item.biotimbre_id})" for item in existing
    )
    message = (
        f"This library already holds this exact recording as {others}, filed "
        f"under a different biotimbre. Two libraries disagree about which "
        f"sound-being it is."
    )
    if policy.duplicate_audio == "skip":
        return "skip", message, ()
    return "add", None, (message + " Importing it anyway, as asked.",)


def _plan_embeddings(
    bundle: Bundle,
    entry: BundleEntry,
    policy: ImportPolicy,
    local_models: Mapping[str, ModelFingerprint],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Decide which of an entry's vectors may be stored as they are.

    A vector is only worth keeping if the recipe behind it matches this
    library's, field for field. Anything else is refused with the reason, so a
    person is told *why* their import ran inference rather than only that it
    did.

    Args:
        bundle: The bundle, for its model fingerprints.
        entry: The entry being planned.
        policy: The choices to apply.
        local_models: This library's registered fingerprints.

    Returns:
        Model ids to reuse, and model id to the reason for each refusal.
    """
    if not entry.embeddings:
        return (), {}
    if not policy.trust_embeddings:
        return (), {
            model_id: "not trusted; recomputing from the audio"
            for model_id in sorted(entry.embeddings)
        }

    reusable: list[str] = []
    rejected: dict[str, str] = {}
    for model_id in sorted(entry.embeddings):
        theirs = bundle.models.get(model_id)
        ours = local_models.get(model_id)
        if ours is None:
            rejected[model_id] = (
                "not registered in this library, so there is no local "
                "dimension to check these vectors against — this library does "
                "not use this model, or its file could not be read"
            )
            continue
        if theirs is None:  # read_bundle refuses this, but do not assume it
            rejected[model_id] = "the bundle gave no fingerprint for this model"
            continue
        differences = theirs.differences(ours)
        if differences:
            rejected[model_id] = (
                f"this library registers {model_id} differently "
                f"({', '.join(differences)}), so the vectors are not comparable"
            )
            continue
        reusable.append(model_id)
    return tuple(reusable), rejected
