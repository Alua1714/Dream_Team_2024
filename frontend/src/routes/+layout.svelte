<script lang="ts">
	import { ModeWatcher } from "mode-watcher";
	import '../app.css';
	let { children } = $props();
</script>

<main class="overflow-hidden">
	{@render children()}
</main>

<ModeWatcher />