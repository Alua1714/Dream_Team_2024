<script lang="ts">
  import { Button } from "$lib/components/ui/button/index.js";
  import { Input } from "$lib/components/ui/input/index.js";
  import { Label } from "$lib/components/ui/label/index.js";

  import * as Card from "$lib/components/ui/card/index.js";
</script>

<Card.Root class="w-[350px]">
  <Card.Header>
    <Card.Title>Import new dataset</Card.Title>
    <Card.Description
      >Upload the file containing the attributes of the houses you want to
      predict the price of.</Card.Description
    >
  </Card.Header>
  <Card.Content>
    <form>
      <div class="grid w-full items-center gap-4">
        <div class="flex flex-col space-y-1.5">
          <Label for="name">Dataset</Label>
          <Input id="picture" type="file" />
        </div>
      </div>
    </form>
  </Card.Content>
  <Card.Footer class="flex justify-between">
    <Button variant="outline">Cancel</Button>
    <Button>Calculate Prices</Button>
  </Card.Footer>
</Card.Root>
