<script lang="ts">
  import { map } from "$lib/state/map.svelte";
  const googleMapsApiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
</script>

<div class="h-screen w-screen">
  <iframe
    title="Google Maps"
    class="h-full w-full border-0 dark:invert-[90%]"
    src={`https://www.google.com/maps/embed/v1/search?key=${googleMapsApiKey}&q=${map.coordinates}`}
  ></iframe>
</div>