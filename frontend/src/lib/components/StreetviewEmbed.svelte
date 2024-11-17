<script lang="ts">
  import { map } from "$lib/state/map.svelte";
  const googleMapsApiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
</script>

<div class="h-80 w-full">
  <iframe
    title="Google Street View"
    class="h-full w-full border-0 dark:invert-[90%]"
    src={`https://www.google.com/maps/embed/v1/streetview?key=${googleMapsApiKey}&location=${map.coordinates}&heading=210&pitch=10&fov=90`}
  ></iframe>
</div>