function createMap() {
  let coordinates = $state("Barcelona");

  function setCoordinates(lat: number, lon: number) {
    coordinates = `${lat},${lon}`;
  }

  return {
    get coordinates() {
      return coordinates;
    },
    setCoordinates,
  };
}

export const map = createMap();
