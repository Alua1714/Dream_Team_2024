<script lang="ts">
  import ModeSwitch from "$lib/components/ModeSwitch.svelte";
  import MapEmbed from "$lib/components/MapEmbed.svelte";
  import { Input } from "$lib/components/ui/input/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import { Label } from "$lib/components/ui/label/index.js";

  import * as Card from "$lib/components/ui/card/index.js";

  let input = $state("");
  let coordinates = $state("Barcelona");

  function handleSubmit() {
    coordinates = input;
  }
</script>

<Card.Root class="absolute top-2 left-2 bottom-16 w-96 z-10">
  <Card.Header>
    <Card.Title>Go to Location</Card.Title>
    <Card.Description>Introduce a Location and automatically navigate to it.</Card.Description>
  </Card.Header>
  <form onsubmit={handleSubmit}>
    <Card.Content>
      <div class="flex flex-col space-y-1.5">
        <Label for="name">Location</Label>
        <Input type="text" bind:value={input} />
      </div>
    </Card.Content>
    <Card.Footer class="mt-auto">
      <Button type="submit" class="w-full">Go to coordinate</Button>
    </Card.Footer>
  </form>
</Card.Root>
 
<div class="absolute top-2 right-2 z-10">
  <ModeSwitch />
</div>

<MapEmbed markers={coordinates} />