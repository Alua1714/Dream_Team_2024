<script lang="ts">
  import { Button } from "$lib/components/ui/button/index.js";
  import { Label } from "$lib/components/ui/label/index.js";
  import { LoaderCircle } from "lucide-svelte";

  import * as Card from "$lib/components/ui/card/index.js";

  let files = $state(null);
  let formLoading = $state(false);

  async function handleSubmit() {
    console.log("Submitting form with files:");
    formLoading = true;
    try {
      const formData = new FormData();
      formData.append('file', files[0]);

      const response = await fetch(`${import.meta.env.VITE_API_ENDPOINT}/upload/`, {
        method: 'POST',
        headers: {
          'accept': 'application/json'
        },
        body: formData
      });
      // success
    } catch (error) {
      // error
    } finally {
      formLoading = false;
    }
  }
</script>

<Card.Root class="w-[350px]">
  <form onsubmit={handleSubmit}>
    <Card.Header>
        <Card.Title>Import new dataset</Card.Title>
        <Card.Description
        >Upload the file containing the attributes of the houses you want to
        predict the price of.</Card.Description
        >
    </Card.Header>
    <Card.Content>
      <div class="grid w-full items-center gap-4">
        <div class="flex flex-col space-y-1.5">
          <Label for="name">Dataset</Label>
          <input type="file" bind:files required>
        </div>
      </div>
  </Card.Content>
  <Card.Footer class="flex justify-between">
    <Button variant="outline">Cancel</Button>
    <Button type="submit" class="w-full" disabled={formLoading}>
      {#if formLoading}
        <LoaderCircle class="h-5 animate-spin" />
      {:else}
        Calculate Prices
      {/if}
    </Button>
  </Card.Footer>
  </form>
</Card.Root>
