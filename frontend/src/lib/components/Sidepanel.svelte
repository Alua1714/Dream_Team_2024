<script lang="ts">
  import { Button } from "$lib/components/ui/button/index.js";
  import { LoaderCircle } from "lucide-svelte";
  import { ScrollArea } from "$lib/components/ui/scroll-area/index.js";
  import InfoDialog from "$lib/components/InfoDialog.svelte";

  import * as Card from "$lib/components/ui/card/index.js";

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

      const response = await fetch(`${import.meta.env.VITE_API_ENDPOINT}/upload/`, {
        method: "POST",
        headers: {
          "accept": "application/json"
        },
        body: formData
      });

      const data = await response.json();
      results = data;
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
    map.setCoordinates(house.location.latitude, house.location.longitude);
    selectedHouse = house;
  }
</script>

<Card.Root class="absolute top-2 left-2 bottom-16 w-96 z-10">
{#if results}
  <div class="px-2 py-6">
    <ScrollArea class="h-[540px] rounded">
      <div class="space-y-3 px-4 w-full">
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
        Upload a file containing the attributes of the houses you want to predict the price of.
      </Card.Description>
    </Card.Header>
    <Card.Content>
      <div class="grid w-full items-center gap-4">
          <div class="relative h-[450px] w-full border-2 border-dashed rounded-lg 
            hover:border-gray-400 transition-colors flex items-center justify-center">
            <input 
                type="file" 
                bind:files 
                required
                class="absolute w-full h-full opacity-0 cursor-pointer"
            />
            <p class="text-sm">Click to upload or drag and drop</p>
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
  class="w-full text-left rounded p-4 transition-all {selectedHouse?.listing_id === house.listing_id ? "border-2 border-primary" : "border"} "
>
  <div>
    <h2>Id: {house.listing_id}</h2>
    <p>Price: {house.prediction}</p>
    <InfoDialog />
  </div>
</button>
{/snippet}