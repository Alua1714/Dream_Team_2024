<script lang="ts">
  import { Button } from "$lib/components/ui/button/index.js";
  import { ScrollArea } from "$lib/components/ui/scroll-area/index.js";
  import { Skeleton } from "$lib/components/ui/skeleton/index.js";

  import * as Card from "$lib/components/ui/card/index.js";

  import { map } from "$lib/state/map.svelte";

  let lon = $state(43.4321);
  let lat = $state(43.4532);

  function selectLocation() {
    map.setCoordinates(lon, lat);
  }

  const skeletons = new Array(20).fill(0);
</script>

<Card.Root class="absolute top-2 left-2 bottom-16 w-96 z-10">
  <div class="py-6 px-2">
    <ScrollArea class="h-[545px] rounded">
        <div class="space-y-3 px-4">
          {#each skeletons as skeleton}
            <Skeleton class="h-20 rounded w-full" />
          {/each}
        </div>
    </ScrollArea>
  </div>
  <Card.Footer class="mt-auto">
    <Button class="w-full" onclick={selectLocation}>Upload Another file</Button>
  </Card.Footer>
</Card.Root>