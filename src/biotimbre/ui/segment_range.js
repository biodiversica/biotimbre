// A two-thumb range whose span cannot exceed maxSpan.
//
// QRange cannot do this: during a drag it re-sends the drag-start value of the
// edge not being dragged on every move, so a cap enforced from the server is
// undone by the next mouse event and the selection flickers between the two.
// Here the cap is applied in the browser, as part of the drag itself: pushing
// an edge past it carries the other edge along, so the whole selection slides.
export default {
  template: `
    <div
      ref="track"
      class="relative w-full select-none"
      style="height: 40px; touch-action: none; cursor: pointer"
      @pointerdown="onDown"
      @pointermove="onMove"
      @pointerup="onUp"
      @pointercancel="onUp"
    >
      <div
        class="absolute w-full rounded"
        style="top: 18px; height: 4px; background: currentColor; opacity: 0.25"
      ></div>
      <div
        class="absolute rounded"
        :style="{
          top: '18px', height: '4px', background: 'var(--q-primary)',
          left: pct(local.min), width: (ratio(local.max) - ratio(local.min)) * 100 + '%',
          cursor: 'grab',
        }"
      ></div>
      <div
        v-for="edge in ['min', 'max']"
        :key="edge"
        class="absolute rounded-full"
        role="slider"
        tabindex="0"
        :aria-valuemin="min"
        :aria-valuemax="max"
        :aria-valuenow="local[edge]"
        :style="{
          top: '10px', width: '20px', height: '20px', marginLeft: '-10px',
          left: pct(local[edge]), background: 'var(--q-primary)',
        }"
        @keydown="onKey(edge, $event)"
      ></div>
    </div>
  `,
  props: {
    modelValue: Object,
    min: Number,
    max: Number,
    step: Number,
    maxSpan: Number,
  },
  data() {
    return { local: { ...this.modelValue }, drag: null };
  },
  watch: {
    modelValue: {
      deep: true,
      handler(value) {
        // The server's echo of an earlier step would pull the thumbs back
        // mid-drag; the drag is the authority until it is released.
        if (!this.drag) this.local = { ...value };
      },
    },
  },
  methods: {
    ratio(value) {
      const width = this.max - this.min;
      return width > 0 ? (value - this.min) / width : 0;
    },
    pct(value) {
      return this.ratio(value) * 100 + "%";
    },
    round(value) {
      const step = this.step || 0;
      const snapped = step > 0 ? Math.round((value - this.min) / step) * step + this.min : value;
      return Math.min(this.max, Math.max(this.min, Number(snapped.toFixed(6))));
    },
    valueAt(event) {
      const box = this.$refs.track.getBoundingClientRect();
      const ratio = box.width > 0 ? (event.clientX - box.left) / box.width : 0;
      return this.min + Math.min(1, Math.max(0, ratio)) * (this.max - this.min);
    },
    // Move one edge to `value`, carrying the other along once the cap is hit.
    moveEdge(edge, value) {
      let { min, max } = this.local;
      const span = this.maxSpan > 0 ? this.maxSpan : this.max - this.min;
      if (edge === "min") {
        min = Math.min(this.round(value), max);
        if (max - min > span) max = this.round(min + span);
      } else {
        max = Math.max(this.round(value), min);
        if (max - min > span) min = this.round(max - span);
      }
      this.update({ min, max });
    },
    update(next) {
      if (next.min === this.local.min && next.max === this.local.max) return;
      this.local = next;
      this.$emit("update:modelValue", { ...next });
    },
    onDown(event) {
      if (event.button !== 0) return;
      const value = this.valueAt(event);
      const box = this.$refs.track.getBoundingClientRect();
      const grab = (12 / Math.max(box.width, 1)) * (this.max - this.min);
      const { min, max } = this.local;
      let type;
      if (Math.abs(value - max) <= grab && value >= min) type = "max";
      else if (Math.abs(value - min) <= grab) type = "min";
      else if (value > min && value < max) type = "range";
      else type = value < min ? "min" : "max";
      this.drag = { type, origin: value, start: { min, max } };
      this.$refs.track.setPointerCapture(event.pointerId);
      if (type !== "range") this.moveEdge(type, value);
    },
    onMove(event) {
      if (!this.drag) return;
      const value = this.valueAt(event);
      if (this.drag.type !== "range") {
        this.moveEdge(this.drag.type, value);
        return;
      }
      const { start, origin } = this.drag;
      const width = start.max - start.min;
      const min = this.round(
        Math.min(this.max - width, Math.max(this.min, start.min + value - origin))
      );
      this.update({ min, max: this.round(min + width) });
    },
    onUp(event) {
      if (!this.drag) return;
      this.drag = null;
      if (this.$refs.track.hasPointerCapture(event.pointerId)) {
        this.$refs.track.releasePointerCapture(event.pointerId);
      }
      this.$emit("change", { ...this.local });
    },
    onKey(edge, event) {
      const keys = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 };
      if (!(event.key in keys)) return;
      event.preventDefault();
      this.moveEdge(edge, this.local[edge] + keys[event.key] * (this.step || 1));
      this.$emit("change", { ...this.local });
    },
  },
};
