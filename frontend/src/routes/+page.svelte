<script lang="ts">
  import ModeSwitch from "$lib/components/ModeSwitch.svelte";
  import MapEmbed from "$lib/components/MapEmbed.svelte";
  import Sidepanel from "$lib/components/Sidepanel.svelte";
</script>

<Sidepanel />

<div class="absolute top-2 right-2 z-10">
  <ModeSwitch />
</div>

<MapEmbed />