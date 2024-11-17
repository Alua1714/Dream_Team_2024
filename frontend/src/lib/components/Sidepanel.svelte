<script lang="ts">
  import { Button } from "$lib/components/ui/button/index.js";
  import { LoaderCircle } from "lucide-svelte";
  import { ScrollArea } from "$lib/components/ui/scroll-area/index.js";

  import * as Card from "$lib/components/ui/card/index.js";
  import MapPin from "lucide-svelte/icons/map-pin";

  import StreetviewEmbed from "$lib/components/StreetviewEmbed.svelte";

  import { map } from "$lib/state/map.svelte";

  let files = $state(null);
  let formLoading = $state(false);
  let results = $state(null);
  let selectedHouse = $state(null);

  async function handleSubmit() {
    formLoading = true;
    try {
      const formData = new FormData();
      formData.append("file", files[0]);

      const response = await fetch(
        `${import.meta.env.VITE_API_ENDPOINT}/upload/`,
        {
          method: "POST",
          headers: {
            accept: "application/json",
          },
          body: formData,
        }
      );

      const data = await response.json();
      results = data;
      selectLocation(results[0])
      // success
    } catch (error) {
      // error
    } finally {
      formLoading = false;
    }
  }

  function resetResults() {
    results = null;
    files = null;
  }

  function selectLocation(house) {
    if (selectedHouse?.listing_id === house.listing_id) {
      selectedHouse = null;
    } else {
      map.setCoordinates(house.location.longitude, house.location.latitude);
      selectedHouse = house;
    }
  }

  function capitalizeWords(str: string) {
    return str
      .split(" ")
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  }
</script>

<Card.Root class="absolute top-2 left-2 bottom-16 w-96 z-10">
  {#if results}
    <div class="px-2 py-6">
      <ScrollArea class="h-[540px] rounded">
        <div class="space-y-4 px-4 w-full">
          {#each results as house}
            {@render houseCard(house)}
          {/each}
        </div>
      </ScrollArea>
    </div>
    <Card.Footer>
      <Button class="w-full" onclick={resetResults}>Upload Another File</Button>
    </Card.Footer>
  {:else}
    <form onsubmit={handleSubmit}>
      <Card.Header>
        <Card.Title>Import New Dataset</Card.Title>
        <Card.Description>
          Upload a file containing the attributes of the houses you want to
          predict the price of.
        </Card.Description>
      </Card.Header>
      <Card.Content>
        <div class="grid w-full items-center gap-4">
          <div
            class="relative h-[450px] w-full border-2 border-dashed rounded-lg
            hover:border-gray-400 transition-colors flex items-center justify-center"
          >
            <input
              type="file"
              bind:files
              required
              class="absolute w-full h-full opacity-0 cursor-pointer"
            />
            {#if files?.[0]}
              <p class="text-sm font-medium">{files[0].name}</p>
            {:else}
              <p class="text-sm">Click to upload or drag and drop</p>
            {/if}
          </div>
        </div>
      </Card.Content>
      <Card.Footer class="flex justify-between">
        <Button type="submit" class="w-full" disabled={formLoading}>
          {#if formLoading}
            <LoaderCircle class="h-5 animate-spin" />
          {:else}
            Calculate Prices
          {/if}
        </Button>
      </Card.Footer>
    </form>
  {/if}
</Card.Root>

{#snippet houseCard(house)}
  <button
    onclick={() => selectLocation(house)}
    class="w-full text-left rounded-lg p-4 pt-5 mt-2 relative border
    {selectedHouse?.listing_id === house.listing_id ? "border-primary shadow" : ""}"
  >
    <div class="flex items-center gap-3">
      <MapPin class="h-5 w-5 text-primary" />
      <h2 class="font-medium text-gray-900">
        {capitalizeWords(house.adress)}
      </h2>
    </div>
    <span class="absolute -top-2 right-3 px-2 bg-red-500 text-white text-sm rounded-full">
      ${Math.round(house.prediction).toLocaleString()}
    </span>
  </button>
{/snippet}

{#if selectedHouse !== null}
  <Card.Root class="absolute right-2 bottom-2 z-10 overflow-hidden shadow-2xl {selectedHouse === null ? "hiden" : ""}">
    <StreetviewEmbed />
    <Card.Header>
      <span class="text-white text-lg bg-red-500 rounded-full px-2 absolute top-3 right-3 shadow-lg">
        ${Math.round(selectedHouse?.prediction).toLocaleString()}
      </span>
      <Card.Title>{selectedHouse?.adress}</Card.Title>
    </Card.Header>
    <Card.Content>
      <div class="text-sm text-gray-500 mb-4">
        <p>Lat: {selectedHouse?.location.latitude}</p>
        <p>Lon: {selectedHouse?.location.longitude}</p>
      </div>
      {#if selectedHouse?.structure_yearbuilt}
        <p>Year Built: {Math.round(selectedHouse.structure_yearbuilt)}</p>
      {/if}
      {#if selectedHouse?.property_propertytype}
        <p>Type: {capitalizeWords(selectedHouse.property_propertytype)}</p>
      {/if}
    </Card.Content>
  </Card.Root>
{/if}
