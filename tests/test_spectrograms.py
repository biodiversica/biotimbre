"""The frequency axis drawn over spectrograms."""

from __future__ import annotations

import pytest

pytest.importorskip("nicegui", reason="needs the 'gui' extra")

from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from biotimbre.ui import spectrograms  # noqa: E402
from biotimbre.ui.spectrograms import (  # noqa: E402
    _kilohertz,
    frequency_ticks,
    marked_spectrogram,
)


class TestTicks:
    def test_steps_are_round_numbers(self):
        for nyquist in (8_000, 11_025, 16_000, 22_050, 24_000, 48_000, 96_000):
            ticks = frequency_ticks(nyquist, 170)
            step = ticks[0]
            # 1, 2 or 5 times a power of ten: values one can type as they are.
            assert str(int(step)).rstrip("0") in {"1", "2", "5"}, (nyquist, step)
            assert ticks == pytest.approx(
                [step * (i + 1) for i in range(len(ticks))]
            )

    def test_ticks_stay_inside_the_image_and_clear_of_its_top(self):
        ticks = frequency_ticks(16_000, 170)
        assert ticks[0] > 0
        assert ticks[-1] <= 16_000 * (1 - spectrograms.TOP_MARGIN)

    def test_a_preview_gets_enough_labels_to_pick_a_cut_off_by(self):
        # The add form's preview is 170 px tall; two labels on a 16 kHz axis
        # would leave the filter cut-offs to guesswork.
        assert len(frequency_ticks(16_000, 170)) >= 5

    def test_a_taller_image_gets_more_labels(self):
        assert len(frequency_ticks(16_000, 290)) >= len(
            frequency_ticks(16_000, 110)
        )

    def test_nothing_to_label_without_a_frequency(self):
        assert frequency_ticks(0, 170) == []

    def test_labels_are_kilohertz_without_trailing_zeros(self):
        assert _kilohertz(2_000) == "2"
        assert _kilohertz(500) == "0.5"
        assert _kilohertz(12_500) == "12.5"


class TestAxis:
    async def test_a_known_rate_labels_the_axis(self, user: User):
        @ui.page("/axis")
        def page() -> None:
            marked_spectrogram(
                "data:,", None, 0.0, 5.0, max_frequency=16_000, height=170
            )

        await user.open("/axis")
        await user.should_see(marker="frequency-axis")
        # The unit rides on the top label rather than on an axis title.
        await user.should_see("14 kHz")

    async def test_no_axis_without_a_rate(self, user: User):
        @ui.page("/noaxis")
        def page() -> None:
            marked_spectrogram("data:,", None, 0.0, 5.0)

        await user.open("/noaxis")
        await user.should_not_see(marker="frequency-axis")


class TestBandShade:
    async def _shaded(self, user: User, path: str, max_frequency=16_000):
        holder: dict = {}

        @ui.page(path)
        def page() -> None:
            holder["shade"] = spectrograms.BandShade()
            marked_spectrogram(
                "data:,",
                None,
                0.0,
                5.0,
                max_frequency=max_frequency,
                height=170,
                band_shade=holder["shade"],
            )

        await user.open(path)
        return holder["shade"]

    async def test_nothing_is_shaded_without_a_filter(self, user: User):
        shade = await self._shaded(user, "/shadenone")
        assert not shade._below.visible
        assert not shade._above.visible

    async def test_a_high_pass_dims_what_lies_below_it(self, user: User):
        shade = await self._shaded(user, "/shadelow")
        shade.place(4_000, None)
        assert shade._below.visible
        assert shade._below._style["height"] == "25.00%"
        assert not shade._above.visible

    async def test_a_low_pass_dims_what_lies_above_it(self, user: User):
        shade = await self._shaded(user, "/shadehigh")
        shade.place(None, 12_000)
        assert shade._above.visible
        assert shade._above._style["height"] == "25.00%"
        assert not shade._below.visible

    async def test_a_low_pass_beyond_nyquist_dims_nothing(self, user: User):
        shade = await self._shaded(user, "/shadebeyond")
        shade.place(None, 20_000)
        assert not shade._above.visible

    async def test_the_band_survives_a_rebuilt_picture(self, user: User):
        holder: dict = {"shade": spectrograms.BandShade()}

        @ui.page("/shaderebuild")
        def page() -> None:
            holder["box"] = ui.column()

        await user.open("/shaderebuild")
        shade = holder["shade"]
        shade.place(4_000, 12_000)
        # As a change of view does: the picture is built again from scratch,
        # and the cut-offs already typed must not have to be typed again.
        with user.client, holder["box"]:
            marked_spectrogram(
                "data:,", None, 0.0, 5.0,
                max_frequency=16_000, height=170, band_shade=shade,
            )
        assert shade._below._style["height"] == "25.00%"
        assert shade._above._style["height"] == "25.00%"
