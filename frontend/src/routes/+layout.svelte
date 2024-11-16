<script lang="ts">
	import Header from "$lib/components/Header.svelte";
	import { ModeWatcher } from "mode-watcher";
	
	import '../app.css';
	let { children } = $props();
</script>

<Header />

<main class="px-6 py-4 mx-auto container">
	{@render children()}
</main>

<ModeWatcher />